from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from tools.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n)) # 创建用于存储固定参数的buffer
        self.register_buffer("bias", torch.zeros(n)) # 
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, 
                              strict, missing_keys, unexpected_keys, 
                              error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, 
                                             local_metadata, strict, missing_keys, 
                                             unexpected_keys, error_msgs)
    
    def forward(self,x):
        w = self.weight.reshape(1,-1,1,1)
        b = self.bias.reshape(1,-1,1,1)
        rv = self.running_var.reshape(1,-1,1,1)
        rm = self.running_mean.reshape(1,-1,1,1)
        eps = 1e-5
        scale = w *(rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
class BackboneBase(nn.Module):  # using model_type to change the structure
    def __init__(self, 
                 backbone:nn.Module, 
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_layers: bool,
                 model_type:str):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # layer0 layer1不需要训练 因为前面层提取的信息其实很有限 都是差不多的 不需要训练
            if not train_backbone or 'layer2'not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
            
        if model_type == 'base':
            if return_interm_layers :
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
                # 检测任务直接返回layer4即可  
                # 执行torchvision.models._utils.IntermediateLayerGetter
                # 这个函数可以直接返回对应层的输出结果
            self.num_channels = num_channels
        elif model_type == 'deformable': # deformable detr系列的模型配置
            print('deformable')
            if return_interm_layers:
                # print('deformable')
                # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # self.num_channels = num_channels
        
    def forward(self, tensorlist:NestedTensor):
        """
        tensor_list: pad预处理之后的图像信息
        tensor_list.tensors: [bs, 3, h, w]预处理后的图片数据 对于小图片而言多余部分用0填充
        tensor_list.mask: [bs, h, w] 用于记录矩阵中哪些地方是填充的（原图部分值为False，填充部分值为True）
        """
        xs = self.body(tensorlist.tensors)
        out:Dict[str, NestedTensor] = {}
        for name,x in xs.items():
            m = tensorlist.mask
            assert m is not None
            # 通过插值函数知道卷积后的特征的mask  知道卷积后的特征哪些是有效的  哪些是无效的
            # 因为之前图片输入网络是整个图片都卷积计算的 生成的新特征其中有很多区域都是无效的
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x,mask)
            # 该循环的主要作用就是通过NestedTensor，确认图像中有效的卷积特征
        return out


class Backbone(BackboneBase):
    """ResNet50 with frozen BatchNorm"""
    def __init__(self,name:str, 
                 train_backbone: bool, 
                 return_interm_layers: bool,
                 dilation:bool,
                 model_type:str):
        backbone = getattr(torchvision.models,name)(
            replace_stride_with_dilation = [False, False, dilation],
            pretrained = is_main_process(), norm_layer=FrozenBatchNorm2d
        ) # is_main_process 的主要作用就是查看分布式训练那边的初始化是否完成
        # DC5表示在主干网络(默认resnet50)的最后一个stage加了个空洞卷积并减少
        # 了个pooling层实现分辨率增大一倍
        num_channels = 512 if name in ('resnet18','resnet34') else 2048
        if model_type == 'deformable' and dilation: # deformable detr在使用空洞卷积的时候会对最后一层进行一次操作
            self.strides[-1] = self.strides[-1] // 2
        super().__init__(backbone, train_backbone, 
                         num_channels, return_interm_layers,model_type)

class Joiner(nn.Sequential):
    # 将ResNet提取的特征与相应的position embedding相结合用于送进后续的Transformer结构进行计算
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
    
    def forward(self,tensor_list:NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name ,x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        
        return out, pos
    
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # 是否需要进行训练
    train_backbone = args.lr_backbone > 0
    # 是否需要返回中间层结果 目标检测False  分割True
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, 
                        return_interm_layers, args.dilation,
                        args.model_type)
    model = Joiner(backbone, position_embedding)
    # model.num_channels = backbone.num_channels
    print(len(backbone.strides))
    return model