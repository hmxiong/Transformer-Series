import copy
from typing import Optional, List

import torch
import math
import torch.nn.functional as F
from torch import nn,Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

# from tools.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from tools.misc import inverse_sigmoid
from .basic_operator import MSDeformAttn

class DeformableTransformer(nn.Module):
    def __init__(self, ffn_hidden_dim=256 , n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward = 1024, dropout = 0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.ffn_hidden_dim = ffn_hidden_dim
        self.n_head = n_head
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(ffn_hidden_dim, dim_feedforward,
                                                          dropout, activation, 
                                                          num_feature_levels, n_head, enc_n_points)
        
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(ffn_hidden_dim, dim_feedforward,
                                                          dropout, activation, 
                                                          num_feature_levels, n_head, dec_n_points)
        
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,return_intermediate_dec)
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, ffn_hidden_dim))

        if two_stage:
            self.enc_output = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
            self.enc_output_norm = nn.LayerNorm(ffn_hidden_dim)
            self.pos_trans = nn.Linear(ffn_hidden_dim * 2, ffn_hidden_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(ffn_hidden_dim * 2)
        else:
            self.reference_points = nn.Linear(ffn_hidden_dim, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)
    
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperture = 10000
        scale = 2*math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperture ** (2 * (dim_t // 2) / num_pos_feats)

        proposals = proposals.sigmoid()*scale

        pos = proposals[:,:,:,None] / dim_t

        pos = torch.stack((pos[:,:,:,0::2].sin(), pos[:,:,:,1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
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
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    # 输入的是多尺度的特征图
    def forward(self, srcs, masks, pos_embeds, query_embed=None):
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
            # bs, h, w, 256 -> h*w, bs, 256
            # 取样点即为特征图的宽高点
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # 将多尺度的信息进行拼接
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

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
        else:
            query_embed , tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            print(query_embed.shape)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        # print(reference_points.shape)
        #decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, 
                                            valid_ratios, query_embed, mask_flatten)
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
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = get_clone(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    # 获取参考点的位置函数
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        # 根据此时输入的特征图的情况确定对应的参考点位置
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H-0.5, H, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W-0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            # print(ref.shape)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # print(reference_points.shape)
        return reference_points
    
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, 
                padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
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
                level_start_index, src_padding_mask=None):
        # sefl attn 
        q = k = self.with_pos_embed(tgt, query_pos)
        # transpose进行矩阵转置，同时只选取其中的有用部分
        tgt2 = self.self_attn(q.transpose(0,1), k.transpose(0,1), tgt.transpose(0,1))[0].transpose(0, 1)
        tgt = tgt+ self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # cross attn
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, 
                 decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = get_clone(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.bbox_embed = None
        self.class_embed = None
    
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios, query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, 
                           src_spatial_shapes, src_level_start_index, src_padding_mask)
            
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
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
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

def get_clone(model, N):
    return nn.ModuleList([copy.deepcopy(model) for i in range(N)])

def build_deforamble_transformer(args):
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
        two_stage_num_proposals=args.num_queries)

