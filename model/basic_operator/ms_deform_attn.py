# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from .ms_deform_attn_func import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        # 采样点的坐标偏移offset
        # 每个query在每个注意力头和每个特征层都需要采样n_points=4个采样点 每个采样点2D坐标 xy = 2  ->  n_heads * n_levels * n_points * 2 = 256
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 每个query对应的所有采样点的注意力权重  n_heads * n_levels * n_points = 8x8x4=128
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # query通过投影得到的value
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # [8 ]  0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # [8 2]
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # [n_heads, n_levels, n_points, xy] = [8, 4, 4, 2]
        # 进行的是中心点偏移量的归一化计算
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # 同一特征层中不同采样点的坐标偏移需要根据特征图的尺度进行一定的尺度
        # 对于第i个采样点，在8个头部和所有特征层中，其坐标偏移为：
        # (i,0) (i,i) (0,i) (-i,i) (-i,0) (-i,-i) (0,-i) (i,-i)   1<= i <= n_points(根据尺度进行变换)
        # 从图形上看，形成的偏移位置相当于3x3正方形卷积核 去除中心 中心是参考点
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 这里设定的是偏移的bias不随计算图计算反向传播
        # 防止偏移产生自动更新，因为对于每个点来说偏移是固定的位置
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        [参数解析]
        query: 4个flatten后的特征图+4个flatten后特征图对应的位置编码 = src_flatten + lvl_pos_embed_flatten
               [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        reference_points: 4个flatten后特征图对应的归一化参考点坐标 每个特征点有4个参考点 xy坐标
                          [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        input_flatten: 4个flatten后的特征图=src_flatten  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        input_spatial_shapes: 4个flatten后特征图的shape [4, 2]
        input_level_start_index: 4个flatten后特征图对应被flatten后的起始索引 [4]  如[0,15100,18900,19850]
        input_padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        # H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64 每张图片不同尺度特征所有的特征点数量
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # [bs  Len_q  256] -> [bs  Len_q  256]
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
        # 根据mask进行相应的掩码操作
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # 把value拆分成8个head      [bs  Len_q  256] -> [bs  Len_q  8  32]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # 计算偏移 [bs  Len_q  256] -> [bs  Len_q  256] ->  [bs  Len_q  8  4  4  2]
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # 根据query计算attn weight [bs  Len_q  256] -> [bs  Len_q  128] -> [bs  Len_q  8  4*4]
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # 每个query在每个注意力头部内，每个特征层都采样4个特征点，即16个采样点(4x4),再对这16个采样点的注意力权重进行softmax（我理解为进行初始化操作）
        # [bs  Len_q  8  16] -> [bs  Len_q  8  16] -> [bs  Len_q  8  4  4]
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        # print(reference_points.shape[-1])
        if reference_points.shape[-1] == 2: # one_stage
            # w h -> h w
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # [bs Len_q 1 n_point 1 2] + [bs Len_q n_head n_level n_point 2] / [1 1 1 n_point 1 2]
            # -> [bs Len_q 1 n_levels n_points 2]
            # 参考点 + 偏移量/特征层宽高 = 采样点    
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4: # two_stage
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        # 输入：采样点位置、注意力权重、所有点的value
        # 具体过程：根据采样点位置从所有点的value中拿出对应的value，并且和对应的注意力权重进行weighted sum
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
