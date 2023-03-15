import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn,Tensor
# 开始构建整个Transformer结构
class Transformer(nn.Module):
    def __init__(self, ffn_hidden_dim = 256,
                 nhead = 8, num_encoder_layers = 6,
                 num_decoder_layers = 6,dim_feedforward = 2048,
                 dropout = 0.1, activation = "relu", 
                 normalization_before = False, return_intermediate_dec = False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(ffn_hidden_dim,nhead,dim_feedforward,
                                                dropout,activation,normalization_before)
        encoder_norm = nn.LayerNorm(ffn_hidden_dim) if normalization_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,encoder_norm)

        decoder_layer = TransformerDecoderLayer(ffn_hidden_dim, nhead,dim_feedforward, 
                                                dropout,activation,normalization_before)

        decoder_norm = nn.LayerNorm(ffn_hidden_dim)
        
        self.decoder = TransformerDecoder(decoder_layer,num_decoder_layers,decoder_norm,return_intermediate_dec) 
        
        self.reset_parameters() # 进行各项参数的初始化

        self.ffn_hidden_dim = ffn_hidden_dim

        self.nhead = nhead

        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, query_embed, pos_embed):
        bs,c,h,w = src.shape
        src = src.flatten(2).permute(2,0,1)
        pos_embed = pos_embed.flatten(2).permute(2,0,1) # flattne N,C,H,W -> HW,N,c
        query_embed = query_embed.unsqueeze(1).repeat(1,bs,1) # 创建新的维度并重复
        mask = mask.flatten(1) # # flattne C,H,W -> C,HW
        # print(pos_embed.shape)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src,src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1,2), memory.permute(1,2,0).view(bs,c,h,w) # 注意的是permute和view的区别


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 编码器连续复制6份
        self.layers = get_clone(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # 内部包括6个编码器，顺序运行
        # src是图像特征输入，shape=hxw,b,256
        output = src
        for layer in self.layers:
            # 第一个编码器输入来自图像特征，后面的编码器输入来自前一个编码器输出
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
        print(src2.shape)
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


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
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
    )