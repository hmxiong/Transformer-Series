import copy
from typing import Optional, List

import torch
import math
import torch.nn.functional as F
from torch import nn,Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from tools.misc import inverse_sigmoid
from .basic_operator import MSDeformAttn
from .transformer import get_clone,gen_sineembed_for_position,MLP

class DeformableTransformer(nn.Module):
    def __init__(self, ffn_hidden_dim=256 , n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward = 1024, dropout = 0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False,
                 use_dn=False):
        super().__init__()

        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_queries = 300
        self.n_head = n_head
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab
        self.use_dn = use_dn

        encoder_layer = DeformableTransformerEncoderLayer(ffn_hidden_dim, dim_feedforward,
                                                          dropout, activation, 
                                                          num_feature_levels, n_head, enc_n_points)
        
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(ffn_hidden_dim, dim_feedforward,
                                                          dropout, activation, 
                                                          num_feature_levels, n_head, dec_n_points)
        
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,return_intermediate_dec,
                                                    use_dab=use_dab, ffn_hidden_dim=ffn_hidden_dim, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)
        
        # 可学习的位置编码[4, 256]，防止出现不同层一样的位置编码，使用可学习参数代替原本的固定参数编码
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, ffn_hidden_dim))

        # 对于
        if two_stage:
            self.enc_output = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
            self.enc_output_norm = nn.LayerNorm(ffn_hidden_dim)
            self.pos_trans = nn.Linear(ffn_hidden_dim * 2, ffn_hidden_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(ffn_hidden_dim * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(ffn_hidden_dim, 2)
        
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            # 服从均匀分布的Glorot初始化器,预防一些参数过大或过小的情况，再保证方差一样的情况下进行缩放，便于计算
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)
    
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        proposals = proposals.sigmoid()*scale

        pos = proposals[:,:,:,None] / dim_t

        pos = torch.stack((pos[:,:,:,0::2].sin(), pos[:,:,:,1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        # 二阶段网络时使用，用于根据Query得到proposals
        N , S , C = memory.shape
        base_scale = 4.0
        proposals = []
        cur = 0
        
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten = memory_padding_mask[:, cur:(cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            # 完成将所有数据的组合
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals
    
    def get_valid_ratio(self, mask):
        # 用于计算不同等级的特征图中含有的ratio
        _, H, W = mask.shape
        # [bs h w] -> [bs 1 1]，通过下面两个公式计算出没有padding的真实特征图宽高
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    # 输入的是多尺度的特征图，以及对应的mask和pos_embedding
    def forward(self, srcs, masks, pos_embeds, query_embed=None, attn_mask=None):
        """
        经过backbone输出4个不同尺度的特征图srcs，以及这4个特征图对应的masks和位置编码
        srcs:  list4  0=[bs 256 H/8 W/8]      1=[bs 256 H/16 W/16] 2=[bs 256 H/32 W/32] 3=[bs 256 H/64 W/64]
        masks: list4  0=[bs H/8 W/8]          1=[bs H/16 W/16]     2=[bs H/32 W/32]     3=[bs H/64 W/64]
        pos_embeds: list4  0=[bs 256 H/8 W/8] 1=[bs 256 H/16 W/16] 2=[bs 256 H/32 W/32] 3=[bs 256 H/64 W/64]
        query_embed: query embedding 参数 [300  512]
        """
        assert self.two_stage or query_embed is not None

        # 为encoder准备输入数据
        # 进行维度变换后的数据
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten= []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            # bs  h  w  256 -> hw  bs  256
            # 取样点即为特征图的宽高点
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # scale-level position embedding  [bs hw c] + [1 1 c] -> [bs hxw c]
            # 每一层所有位置加上相同的level_embed 且 不同层的level_embed不同
            # 所以这里pos_embed + level_embed，这样即使不同层特征有相同的w和h，那么也会产生不同的lvl_pos_embed
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            # 得到的flatten维度为lvl hw  bs  256
            src_flatten.append(src)
            mask_flatten.append(mask)
        # 将多尺度的信息进行拼接
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64  256]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # 为decoder准备数据
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        elif self.use_dab:
            reference_points = query_embed[..., self.ffn_hidden_dim:].sigmoid() 
            tgt = query_embed[..., :self.ffn_hidden_dim]
            if self.use_dn == False:
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed , tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # print(query_embed.shape)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        # print(reference_points.shape)
        # decoder
        # tgt: 初始化query embedding [bs  300  256]
        # reference_points: 由query pos接一个全连接层 再归一化后的参考点中心坐标 [bs  300  2]
        # query_embed: query pos[bs, 300, 256]
        # memory: Encoder输出结果 [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64  256]
        # spatial_shapes: [4  2] 4个特征层的shape
        # level_start_index: [4  ] 4个特征层flatten后的开始index
        # valid_ratios: [bs  4  2]
        # mask_flatten: 4个特征层flatten后的mask [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64]
        # hs: 6层decoder输出 [n_decoder bs num_query ffn_hidden_dim] = [6 bs 300 256]
        # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6 bs 300 2]
        #                   one-stage=[n_decoder bs num_query 2]  two-stage=[n_decoder bs num_query 4]
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, 
                                            valid_ratios, 
                                            query_pos=query_embed if not self.use_dab else None,
                                            src_padding_mask=mask_flatten, attn_mask = attn_mask)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None
    
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 ffn_hidden_dim = 256, d_ffn=1024,
                 dropout=0.1, activation='relu',
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(ffn_hidden_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(ffn_hidden_dim)

        self.linear1 = nn.Linear(ffn_hidden_dim, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, ffn_hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ffn_hidden_dim)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self,src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        # reference_points 代表的就是相应的参考偏移点，论文中取4
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, 
                              spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # fead_forward_network
        # hw,b,256 -> hw,b,1024 -> hw, b, 256
        # print(src.shape)
        src = self.forward_ffn(src)
        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, model_type):
        super().__init__()
        self.layers = get_clone(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.model_type = model_type

    @staticmethod
    # 获取参考点的位置函数
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        # 根据此时输入的特征图的情况确定对应的参考点位置
        # ref_y: [H, W]  第一行：W个0.5  第二行：W个1.5 ... 第H行：W个99.5
        # ref_x: [H, W]  第一行：0.5 1.5...149.5   H行全部相同
        for lvl, (H, W) in enumerate(spatial_shapes):
            # 根据的输入特征图的形状产生均匀离散的点
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H-0.5, H, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W-0.5, W, dtype=torch.float32, device=device))
            # [H, W] -> [bs, HW]  W个0.5 + W个1.5 + ... + W个99.5 -> 除以H 归一化
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            # [bs  HW  2] 每一项都是xy
            ref = torch.stack((ref_x, ref_y), -1)
            # print(ref.shape)
            reference_points_list.append(ref)
        # [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64  2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points: [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64  2] -> [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64  1  2]
        # valid_ratios: [1  4  2] -> [1  1  4  2]
        # 复制4份 每个特征点都有4个归一化参考点 -> [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64  4  2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # print(reference_points.shape)
        return reference_points
    
    def forward(self, src, spatial_shapes, level_start_index, 
                valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        # 实际上这一部分是不需要的
        intermediate_output = []
        intermediate_ref = []
       
            
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, padding_mask)
            
        return output

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, ffn_hidden_dim=256,
                 d_ffn=1024, dropout=0.1,activation='relu',
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.cross_attn = MSDeformAttn(ffn_hidden_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(ffn_hidden_dim)

        self.self_attn = nn.MultiheadAttention(ffn_hidden_dim, n_heads, dropout = dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ffn_hidden_dim)

        self.linear1 = nn.Linear(ffn_hidden_dim, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, ffn_hidden_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(ffn_hidden_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,
                level_start_index, src_padding_mask=None, self_attn_mask=None):
        # sefl attn 
        q = k = self.with_pos_embed(tgt, query_pos)
        # transpose进行矩阵转置，同时只选取其中的有用部分
        # self-attention
        # 第一个attention的目的：学习各个物体之间的关系/位置   可以知道图像当中哪些位置会存在物体  物体信息->tgt
        # 所以qk都是query embedding + query pos   v就是query embedding
        tgt2 = self.self_attn(q.transpose(0,1), k.transpose(0,1), tgt.transpose(0,1), attn_mask=self_attn_mask)[0].transpose(0, 1)
        tgt = tgt+ self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # cross attention  使用（多尺度）可变形注意力模块替代原生的Transformer交叉注意力
        # 第二个attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中去，更好地表征了图像中的各个物体
        # 所以q=query embedding + query pos, k = query pos通过一个全连接层->2维, v=上一层输出的output
        # cross attn
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, 
                 use_dab=False,ffn_hidden_dim=256, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()
        self.layers = get_clone(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.bbox_embed = None
        self.class_embed = None

        self.use_dab = use_dab
        self.ffn_hidden_dim = ffn_hidden_dim
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(ffn_hidden_dim, ffn_hidden_dim, ffn_hidden_dim, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, ffn_hidden_dim, ffn_hidden_dim, 3)
            else:
                self.ref_point_head = MLP(2*ffn_hidden_dim, ffn_hidden_dim, ffn_hidden_dim, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(ffn_hidden_dim, ffn_hidden_dim, ffn_hidden_dim, 2)
    
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios, query_pos=None, src_padding_mask=None, attn_mask=None):
        """
        tgt: 预设的query embedding [bs  300  256]
        query_pos: 预设的query pos [bs  300  256]
        reference_points: query pos通过一个全连接层->2维  [bs  300  2]
        src: encoder最后的输出特征 即memory [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64 256]
        src_spatial_shapes: [4 2] 4个特征层的原始shape
        src_level_start_index: [4,] 4个特征层flatten后的开始index
        src_padding_mask: 4个特征层flatten后的mask [bs  H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64]
        """
        if self.use_dab:
            assert query_pos is None
            if attn_mask is not None:
                bs = src.shape[0]
                reference_points = reference_points[None].repeat(bs, 1, 1) # bs, nq, 4(xywh)
        
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # [bs 300 1 2] * [bs 1 4 2] -> [bs 300 4 2]=[bs n_query n_lvl 2]
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.use_dab:
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:,:,0,:])
                    raw_query_pos = self.ref_point_head(query_sine_embed)
                pos_scale = self.query_scale(output) if lid != 0 else 1 # 重点， 只在第一层进行一次scale
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)
            # decoder layer
            # output: [bs 300 256] = self-attention输出特征 + cross-attention输出特征
            # 知道图像中物体与物体之间的关系 + encoder增强后的图像特征 + 图像与物体之间的关系
            output = layer(output, query_pos, reference_points_input, src, 
                           src_spatial_shapes, src_level_start_index, 
                           src_padding_mask, self_attn_mask=attn_mask)
            
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            
            # 默认返回6个decoder层输出一起计算损失
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            # 0 [6 bs 300 256] 6层decoder输出
            # 1 [6 bs 300 2] 6层decoder的参考点归一化中心坐标  一般6层是相同的
            # 但是如果是iterative bounding box refinement会不断学习迭代得到新的参考点 6层一半不同
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu , not{activation}.")

def build_deformable_transformer(args):
    return DeformableTransformer(
        ffn_hidden_dim=args.hidden_dim,
        n_head=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        use_dab=args.use_dab,
        use_dn=args.use_dn)

