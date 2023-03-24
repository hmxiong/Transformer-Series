import math
import torch
from torch import nn

from tools.misc import NestedTensor

class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats # 需要进行向量编码的维度 64
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be Ture if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list:NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        # 附加的mask，shape是B,h,w 全是false
        not_mask = ~mask
        # 以第2维度的数据为基础累加上去，行不动,B,h,w，用于记录x轴的坐标
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 以第3维度的数据为基础累加上去，列不动,B,h,w， 用于记录y轴的坐标
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # 取每组数据的最后一个进行归一化计算
            y_embed = (y_embed - 0.5) / (y_embed[:,-1:,:] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:,:,-1:] + eps) * self.scale
        # 对应的就是公式上面的时间变化程度 64
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device =x.device)
        # 2i/2i+1: 2 * (dim_t // 2)  self.temperature=10000   self.num_pos_feats = d/2
        dim_t = self.temperature ** (2 * (dim_t // 2) /self.num_pos_feats)
        # B,h,w,128
        pos_x = x_embed[:,:,:,None] / dim_t
        pos_y = y_embed[:,:,:,None] / dim_t
        # x方向位置编码: [bs,19,26,64][bs,19,26,64] -> [bs,19,26,64,2] -> [bs,19,26,128]
        pos_x = torch.stack((pos_x[:,:,:,0::2].sin(), pos_x[:,:,:,1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:,:,:,0::2].sin(), pos_y[:,:,:,1::2].cos()), dim=4).flatten(3)
        # B,128,h,w flatten(x)以x维度开始进行展平
        # concat: [bs,h,w,128][bs,h,w,128] -> [bs,h,w,256] -> [bs,256,h,w]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # B,256,h,w
        return pos

class PositionEmbeddingSineHW(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        # 通过这种方式记录x和y轴的坐标情况
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # import ipdb; ipdb.set_trace()
        
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        
        # 通过temperatureW 和temperatureH能够对于x y轴的坐标设置不同的温度系数从而改变相应的位置编码情况
        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # import ipdb; ipdb.set_trace()

        return pos

class PositionEmbeddingLearned(nn.Module):
    def __init__(self,num_pos_feats=256):
        super().__init__()
        # 设置的是embedding层的情况来进行相应的计算？
        # # 一个50个单词，每个单词256个字母的字典
        self.row_embed = nn.Embedding(50, num_pos_feats) # 50 * 256
        self.col_embed = nn.Embedding(50, num_pos_feats) # 50 * 256
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list:NestedTensor):
        x = tensor_list.tensors # B,h,w
        h,w = x.shape[-2:]
        i = torch.arange(w, device = x.device) # w
        j = torch.arange(h, device = x.device) # h
        x_emb = self.col_embed(i) # 50 * w
        y_emb = self.row_embed(j) # 50 * h
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h,1,1), # 如何理解repeat函数,整体用于完成重复任务
            y_emb.unsqueeze(1).repeat(1,w,1),
        ],dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)
        return pos

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding in ('v4', 'dab'):
        position_embedding = PositionEmbeddingSineHW(
                N_steps, 
                temperatureH=args.pe_temperatureH,
                temperatureW=args.pe_temperatureW,
                normalize=True
            )
    else :
        raise ValueError(f"not suppoorted {args.position_embedding}")
    
    return position_embedding