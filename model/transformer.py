import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn,Tensor
from tools.misc import inverse_sigmoid
from .attention import MultiheadAttention
# from .detr import (MLP)
class MLP(nn.Module):
    """ a simple mlp perception (FeedForwardNetwork)"""
    def __init__(self, input_dim, hidden_dim,output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n , k in zip([input_dim] + h, h+ [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# 开始构建整个Transformer结构
class Transformer(nn.Module):
    def __init__(self, ffn_hidden_dim = 256,
                 nhead = 8, num_encoder_layers = 6,
                 num_decoder_layers = 6,dim_feedforward = 2048,
                 dropout = 0.1, activation = "relu", 
                 normalization_before = False, return_intermediate_dec = False,
                 model_type = 'base',
                 query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 num_queries=300,
                 use_dn=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(ffn_hidden_dim,nhead,dim_feedforward,
                                                dropout,activation,normalization_before)
        encoder_norm = nn.LayerNorm(ffn_hidden_dim) if normalization_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,encoder_norm,
                                          ffn_hidden_dim=ffn_hidden_dim, model_type=model_type)

        decoder_norm = nn.LayerNorm(ffn_hidden_dim)
        
        if model_type == 'base':
            decoder_layer = TransformerDecoderLayer(ffn_hidden_dim, nhead,dim_feedforward, 
                                                    dropout,activation,normalization_before)
            
            self.decoder = TransformerDecoder(decoder_layer,num_decoder_layers,decoder_norm,
                                            return_intermediate_dec) ### has change
        
        elif model_type == 'conditional':
            decoder_layer = ConditionalTransformerDecoderLayer(ffn_hidden_dim, nhead, dim_feedforward,
                                                dropout, activation, normalization_before)

            self.decoder = ConditionalTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec,
                                            ffn_hidden_dim=ffn_hidden_dim)
        elif model_type == 'dab':
            decoder_layer = DABTransformerDecoderLayer(ffn_hidden_dim, nhead, dim_feedforward,
                                                dropout, activation, normalization_before,
                                                keep_query_pos=keep_query_pos)

            self.decoder = DABTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          ffn_hidden_dim=ffn_hidden_dim, query_dim=query_dim, 
                                          keep_query_pos=keep_query_pos, 
                                          query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)
            
            assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.reset_parameters() # 进行各项参数的初始化

        self.ffn_hidden_dim = ffn_hidden_dim

        self.nhead = nhead

        self.dec_layers = num_decoder_layers # conditional detr
        
        self.model_type = model_type

        # DeNoisng 
        self.use_dn = use_dn
        
        # DAB module
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, ffn_hidden_dim)

        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, query_embed, pos_embed, tgt=None, attn_mask=None):
        bs,c,h,w = src.shape
        src = src.flatten(2).permute(2,0,1)
        pos_embed = pos_embed.flatten(2).permute(2,0,1) # flattne N,C,H,W -> HW,N,c
        # query_embed一个可学习的embedding层，作为另外一种位置编码来理解
        if not self.use_dn:
            query_embed = query_embed.unsqueeze(1).repeat(1,bs,1) # 创建新的维度并重复 100 256 -> 100 bs 256
            tgt = torch.zeros_like(query_embed)
        else:
            refpoint_embed = query_embed
        mask = mask.flatten(1) # # flattne C,H,W -> C,HW
        # print(pos_embed.shape)
        # print(query_embed.shape)

        # 一个和embedding维度保持一致的输入tgt
        # tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src,src_key_padding_mask=mask, pos=pos_embed)
        if self.model_type == 'base':
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                            pos=pos_embed, query_pos=query_embed)

            return hs.transpose(1,2), memory.permute(1,2,0).view(bs,c,h,w) # 注意的是permute和view的区别
        elif self.model_type == 'conditional':
            hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
            return hs, references
        elif self.model_type == 'dab':
            # 确定是否基于DAB进行DeNoising处理来进行decoder计算
            if self.use_dn:
                if self.num_patterns > 0:
                    l = tgt.shape[0]
                    print(tgt.shape)
                    tgt[l - self.num_queries * self.num_patterns:] += \
                        self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)
                # print(tgt.shape)
                hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
                                pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
            else:
                num_queries = query_embed.shape[0]
                # print(query_embed.shape)
                if self.num_patterns == 0:
                    tgt = torch.zeros(num_queries, bs, self.ffn_hidden_dim, device=query_embed.device)
                else:
                    tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # n_q*n_pat, bs, ffn_hidden_dim
                    query_embed = query_embed.repeat(self.num_patterns, 1, 1) # n_q*n_pat, bs, ffn_hidden_dim
                    # import ipdb; ipdb.set_trace()
                hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                pos=pos_embed, refpoints_unsigmoid=query_embed)
            return hs, references

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, ffn_hidden_dim=256, model_type=None):
        super().__init__()
        # 编码器连续复制6份
        self.layers = get_clone(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.model_type = model_type
        if model_type in ['dab']:
            self.query_scale = MLP(ffn_hidden_dim, ffn_hidden_dim, ffn_hidden_dim, 2)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # 内部包括6个编码器，顺序运行
        # src是图像特征输入，shape=hw b 256
        output = src
        for layer in self.layers:
            if self.model_type in ['dab']:
                pos_scales = self.query_scale(output)
                # 第一个编码器输入来自图像特征，后面的编码器输入来自前一个编码器输出
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)
            else:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layers, num_layers ,norm = None, return_intermediate = False):
        super().__init__()
        self.layers = get_clone(decoder_layers, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
    #值得注意的是：在使用TransformerDecoder时需要传入的参数有：
    # tgt：Decoder的输入，memory：Encoder的输出，pos：Encoder的位置编码的输出，query_pos：Object Queries，一堆mask
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        # Decoder输入的tgt:(100, b, 256)
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, ffn_hidden_dim, nhead, dim_feedforward = 2048,
                 drop_out = 0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(ffn_hidden_dim, nhead, dropout=drop_out)
        self.linear1 = nn.Linear(ffn_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(dim_feedforward, ffn_hidden_dim)

        self.norm1 = nn.LayerNorm(ffn_hidden_dim)
        self.norm2 = nn.LayerNorm(ffn_hidden_dim)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self,tensor, pos:Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_post(self,
                     src,
                     src_mask:Optional[Tensor] = None,
                     src_key_padding_mask:Optional[Tensor] = None,
                     pos:Optional[Tensor] = None
                    ):
        # 此处为DETR中Encoder结构的主体部分，注意的是q和k的来源是不一样的 
        q = k = self.with_pos_embed(src, pos)
        # q = k = v = hw,b,256
        src2 = self.self_attn(q,k,value = src, attn_mask = src_mask,
                              key_padding_mask = src_key_padding_mask)[0]
        # print(src2.shape)
        # self_attn = hw,b,256
        # encoderlayer 中的连接部分
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 得到的是FFN网络 + DP + Activation
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # hw,b,256 -> hw,b,2048(multi_head_fusion) -> hw,b,256
        # print(self.linear1(src).shape)
        src = src + self.dropout2(src2)
        # 得到的是最终的输出
        src = self.norm2(src)
        return src
    
    def forward_pre(self,
                    src,
                    src_mask:Optional[Tensor] = None,
                    src_key_padding_mask:Optional[Tensor] = None,
                    pos:Optional[Tensor] = None
                   ):
        # 这部分的代码主要的差异就是在进行Normalization的时间上面有差异    
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q,k,value = src2, attn_mask = src_mask,
                              key_padding_mask = src_key_padding_mask)[0]
        # encoderlayer 中的连接部分
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    
    def forward(self,
                src,
                src_mask:Optional[Tensor] = None,
                src_key_padding_mask:Optional[Tensor] = None,
                pos:Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, ffn_hidden_dim, nhead, dim_feedforward = 2048,
                 drop_out = 0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(ffn_hidden_dim, nhead, dropout=drop_out)
        self.multihead_attn = nn.MultiheadAttention(ffn_hidden_dim, nhead, dropout=drop_out)
        self.linear1 = nn.Linear(ffn_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(dim_feedforward, ffn_hidden_dim)

        self.norm1 = nn.LayerNorm(ffn_hidden_dim)
        self.norm2 = nn.LayerNorm(ffn_hidden_dim)
        self.norm3 = nn.LayerNorm(ffn_hidden_dim)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self,tensor, pos:Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask:Optional[Tensor] = None,
                     memory_mask:Optional[Tensor] = None,
                     tgt_key_padding_mask:Optional[Tensor] = None,
                     memory_key_padding_mask:Optional[Tensor] = None,
                     pos:Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None
                    ):
        # 此处为DETR中Decoder结构的主体部分，注意的是q和k的来源是不一样的 
        # query,key的输入是object queries(query_pos) + Decoder的输入(tgt),shape都是(100,b,256)
        # value的输入是Decoder的输入(tgt),shape = (100,b,256)
        # 初始的tgt是全零的向量参数， object queries在每个decoder中间都会要加上去
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q,k,value = tgt, attn_mask = tgt_mask,
                              key_padding_mask = tgt_key_padding_mask)[0]
        # decoderlayer 中的连接部分
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 进行的是内部的自注意力机制的计算，这个地方qkv的来源要严格对其
        # query的输入是上一个attention的输出(tgt) + object queries(query_pos)
        # key的输入是Encoder的位置编码(pos) + Encoder的输出(memory)
        # value的输入是Encoder的输出(memory)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # 同样的主要改变就是调整了一下进行normalization的位置
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class ConditionalTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, ffn_hidden_dim=256):
        super().__init__()
        self.layers = get_clone(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(ffn_hidden_dim, ffn_hidden_dim, ffn_hidden_dim, 2)
        self.ref_point_head = MLP(ffn_hidden_dim, ffn_hidden_dim, 2, 2)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        # 将原始的256维的信息转为一个2D的中心点位置信息进行后续的修正
        reference_points_before_sigmoid = self.ref_point_head(query_pos)    # [num_queries, batch_size, 2]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for layer_id, layer in enumerate(self.layers):
            # [num_queries, batch_size, 2]
            obj_center = reference_points[..., :2].transpose(0, 1)      

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)

            # get sine embedding for the query vector
            # 
            query_sine_embed = gen_sineembed_for_position(obj_center)     
            # apply transformation
            # 自己理解为进行的是一个候选框位置的选择(pos_transformation)
            query_sine_embed = query_sine_embed * pos_transformation
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]

        return output.unsqueeze(0)


class ConditionalTransformerDecoderLayer(nn.Module):
    def __init__(self, ffn_hidden_dim, n_head, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalization_before=False):
        super().__init__()
        #  Decoder Self-Attn
        # 对于输入的每个object query来说是要进行相应的线性投影然后再进入atten进行计算
        self.sa_qcontent_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.sa_qpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.sa_kcontent_proj = nn.Linear(ffn_hidden_dim , ffn_hidden_dim)
        self.sa_kpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.sa_v_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.self_attn = MultiheadAttention(ffn_hidden_dim, n_head, dropout= dropout, vdim=ffn_hidden_dim)
        
        # Decoder Cross-Attn
        self.ca_qcontent_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_qpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_kcontent_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_kpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_v_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_qpos_sine_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.cross_attn = MultiheadAttention(ffn_hidden_dim*2, n_head, dropout=dropout, vdim=ffn_hidden_dim)

        self.nhead = n_head

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(ffn_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, ffn_hidden_dim)

        self.norm1 = nn.LayerNorm(ffn_hidden_dim)
        self.norm2 = nn.LayerNorm(ffn_hidden_dim)
        self.norm3 = nn.LayerNorm(ffn_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalization_before
    
    def with_pos_embed(self, tensor, pos:Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        # 做一次projection
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        # 数据解耦
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        # 将数据按照head进行分离再融合到一起完成整体的拼接
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)# hw bs 512
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2) # hw bs 512
        # 所有的q k的数据维度都是保持一致，但是和v不一样

        tgt2 = self.cross_attn(query=q,
                                key=k,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        if self.normalize_before:
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)


class DABTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm= None, return_intermediate=False,
                ffn_hidden_dim=256, query_dim=2, keep_query_pos=False, 
                query_scale_type='cond_elewise',
                modulate_hw_attn=False, bbox_embed_diff_each_layer= False):
        super().__init__()
        self.layers = get_clone(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm 
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        # 定义进行框选择的方法和注意事项 
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(ffn_hidden_dim, ffn_hidden_dim, ffn_hidden_dim, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(ffn_hidden_dim, ffn_hidden_dim, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, ffn_hidden_dim)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        # 参考点的特征解耦头
        # input dim = 4 // 2 * 256
        self.ref_point_head = MLP(query_dim // 2 * ffn_hidden_dim , ffn_hidden_dim, ffn_hidden_dim, 2)

        self.bbox_embed = None
        self.ffn_hidden_dim = ffn_hidden_dim
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        
        # 如果需要进行宽高调制注意力机制则需要改变输出与anchor之间的MLP的输出维度
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(ffn_hidden_dim, ffn_hidden_dim, 2, 2)
        
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None
        
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] =None,
                refpoints_unsigmoid: Optional[Tensor] = None): # # num_queries, bs, 4
        
        output = tgt

        intermediate = []
        # 参考点需要进行一次sigmoid归一化处理
        reference_points = refpoints_unsigmoid.sigmoid()
        # num_queries, bs, 4
        # print(reference_points.shape)
        ref_points = [reference_points]
        
        for layer_id , layer in enumerate(self.layers):
            #  num_query bs 4
            obj_center = reference_points[..., :self.query_dim]
            # num_query bs 512
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # print(query_sine_embed.shape)
            # num_query bs 256
            query_pos = self.ref_point_head(query_sine_embed)

            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            
            query_sine_embed = query_sine_embed[..., :self.ffn_hidden_dim] * pos_transformation
            
            # 宽高调制引入了宽高位置信息query_sine_embed此时已经融入了宽高位置信息
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()
                query_sine_embed[..., self.ffn_hidden_dim // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.ffn_hidden_dim // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
            # num_query bs 256
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))
            # print(output.shape)
            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)




class DABTransformerDecoderLayer(nn.Module):
    def __init__(self, ffn_hidden_dim, n_head, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalization_before=False,
                 keep_query_pos=False, rm_self_attn_decoder=False):
        super().__init__()
        #  Decoder Self-Attn
        if not rm_self_attn_decoder:
            # 对于输入的每个object query来说是要进行相应的线性投影然后再进入atten进行计算
            self.sa_qcontent_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
            self.sa_qpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
            self.sa_kcontent_proj = nn.Linear(ffn_hidden_dim , ffn_hidden_dim)
            self.sa_kpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
            self.sa_v_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
            self.self_attn = MultiheadAttention(ffn_hidden_dim, n_head, dropout= dropout, vdim=ffn_hidden_dim)
            
            self.norm1 = nn.LayerNorm(ffn_hidden_dim)
            self.dropout1 = nn.Dropout(dropout)
        # Decoder Cross-Attn
        self.ca_qcontent_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_qpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_kcontent_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_kpos_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_v_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.ca_qpos_sine_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.cross_attn = MultiheadAttention(ffn_hidden_dim*2, n_head, dropout=dropout, vdim=ffn_hidden_dim)

        self.nhead = n_head
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(ffn_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, ffn_hidden_dim)

        self.norm2 = nn.LayerNorm(ffn_hidden_dim)
        self.norm3 = nn.LayerNorm(ffn_hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.keep_query_pos = keep_query_pos
    
    def with_pos_embed(self, tensor, pos:Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        # 做一次projection
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        # 将数据按照head进行分离再融合到一起完成整体的拼接
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)# hw bs 512
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2) # hw bs 512
        # 所有的q k v的数据维度都是保持一致

        tgt2 = self.cross_attn(query=q,
                                key=k,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

        

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":  # when using DAB thia has to change
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu , not{activation}.")

def get_clone(model, N):
    return nn.ModuleList([copy.deepcopy(model) for i in range(N)])


def build_transformer(args):
    return Transformer(
        ffn_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        nhead = args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers = args.enc_layers,
        num_decoder_layers = args.dec_layers,
        normalization_before=args.pre_norm,
        return_intermediate_dec=True,
        model_type = args.model_type,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        use_dn = args.use_dn, 
    )