# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
import math
from pathlib import Path
from PIL import Image
import requests
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tools.box_ops import box_cxcywh_to_xyxy, rescale_bboxes
import torchvision.transforms as T
from model import build_model
from model.position_encoding import PositionEmbeddingSine
from torch import nn
from torchvision.models import resnet50
from torch.nn.functional import dropout,linear,softmax
torch.set_grad_enabled(False)

def get_args_parser():
    parser = argparse.ArgumentParser('DETR test model script', add_help=False)
    parser.add_argument('--model_type','-m', default='base', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned','dab'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # when using deformable is 1024
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, # when using dab is 0
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,   # when using deformable and conitional is 300
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Variants of DAB DETR
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, # setting the tempperture
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")
    parser.add_argument('--num_select', default=100, type=int,  # when using DAB this is 300 and when using dab deformabel is 100
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, 
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")
    parser.add_argument('--use_dab', default=False, action='store_true')
    
    # Variants of DN Options
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int,
                        help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")
    parser.add_argument('--contrastive', action="store_true",
                        help="use contrastive training.")
    parser.add_argument('--use_mqs', action="store_true",
                        help="use mixed query selection from DINO.")
    parser.add_argument('--use_lft', action="store_true",
                        help="use look forward twice from DINO.")


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float) # when using dab is 1 deformabel is 2
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args): 
    # a = np.ones([2, 256, 228, 228], dtype=float)
    # d = torch.tensor(a)
    # position_embedding = PositionEmbeddingSine(256 // 2, normalize=True)
    # out = position_embedding(d)
    # print(out.shape)
    CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
     ]
    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
            T.Resize([800], max_size=1333),
            normalize,
        ])


    # 加载线上的模型
    model, criterion, postprocessors = build_model(args)
    # model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()
    # 获取训练好的参数
    for name, parameters in model.named_parameters():
        # 获取训练好的object queries，即pq:[100,256]
        if name == 'query_embed.weight':
            pq = parameters
        # 获取解码器的最后一层的交叉注意力模块中q和k的线性权重和偏置:[256*3,256]，[768]
        if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_weight':
            in_proj_weight = parameters
        if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_bias':
            in_proj_bias = parameters
    # 线上下载图像
    img_path = '/ssd1/lipengxiang/hmxiong/Transformer-Series/pics/test.jpg'
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    cq = []     # 存储detr中的 cq
    pk =  []    # 存储detr中的 encoder pos
    memory = [] # 存储encoder的输出特征图memory

    # 注册hook
    hooks = [
        # 获取resnet最后一层特征图
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        # 获取encoder的图像特征图memory
        model.transformer.encoder.register_forward_hook(
            lambda self, input, output: memory.append(output)
        ),
        # 获取encoder的最后一层layer的self-attn weights
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        # 获取decoder的最后一层layer中交叉注意力的 weights
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
        # 获取decoder最后一层self-attn的输出cq
        model.transformer.decoder.layers[-1].norm1.register_forward_hook(
            lambda self, input, output: cq.append(output)
        ),
        # 获取图像特征图的位置编码pk
        model.backbone[-1].register_forward_hook(
            lambda self, input, output: pk.append(output)
        ),
    ]

    # propagate through the model
    outputs = model(img)

    # 用完的hook后删除
    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]       # [1,2048,25,34]
    enc_attn_weights = enc_attn_weights[0] # [1,850,850]   : [N,L,S]
    dec_attn_weights = dec_attn_weights[0] # [1,100,850]   : [N,L,S] --> [batch, tgt_len, src_len]
    memory = memory[0] # [850,1,256]

    cq = cq[0]    # decoder的self_attn:最后一层输出[100,1,256]
    pk = pk[0]    # [1,256,25,34]

    # 绘制postion embedding
    pk = pk.flatten(-2).permute(2,0,1)           # [1,256,850] --> [850,1,256]
    pq = pq.unsqueeze(1).repeat(1,1,1)           # [100,1,256]
    q = pq + cq
    #------------------------------------------------------#
    #   1) k = pk，则可视化： (cq + oq)*pk
    #   2_ k = pk + memory，则可视化 (cq + oq)*(memory + pk)
    #   读者可自行尝试
    #------------------------------------------------------#
    k = pk
    # k = pk + memory
    #------------------------------------------------------#

    # 将q和k完成线性层的映射，代码参考自nn.MultiHeadAttn()
    _b = in_proj_bias
    _start = 0
    _end = 256
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q = linear(q, _w, _b)

    _b = in_proj_bias
    _start = 256
    _end = 256 * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k = linear(k, _w, _b)

    scaling = float(256) ** -0.5
    q = q * scaling
    q = q.contiguous().view(100, 8, 32).transpose(0, 1)
    k = k.contiguous().view(-1, 8, 32).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    print(attn_output_weights.shape)
    attn_output_weights = attn_output_weights.view(1, 8, 100, 850)
    attn_output_weights = attn_output_weights.view(1 * 8, 100, 850)
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = attn_output_weights.view(1, 8, 100, 850)

    # 后续可视化各个头
    attn_every_heads = attn_output_weights # [1,8,100,850]
    attn_output_weights = attn_output_weights.sum(dim=1) / 8 # [1,100,850]

    #-----------#
    #   可视化
    #-----------#
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=10, figsize=(22, 28))  # [11,2]
    colors = COLORS * 100

    # 可视化
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        # 可视化decoder的注意力权重
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}',fontsize = 30)
        # 可视化框和类别
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()],fontsize = 30)
        # 分别可视化8个头部的位置特征图
        for head in range(2, 2 + 8):
            ax = ax_i[head]
            ax.imshow(attn_every_heads[0, head-2, idx].view(h,w))
            ax.axis('off')
            ax.set_title(f'head:{head-2}',fontsize = 30)
    fig.tight_layout()        # 自动调整子图来使其填充整个画布
    plt.savefig('/ssd1/lipengxiang/hmxiong/Transformer-Series/pics/savefig_example.jpg')
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR test model script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
