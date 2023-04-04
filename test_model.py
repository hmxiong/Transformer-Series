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
from IPython.display import display, clear_output

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, DistributedSampler
from tools.box_ops import box_cxcywh_to_xyxy, rescale_bboxes
import torchvision.transforms as T
import torch.nn.functional as F
from model import build_model
from model.position_encoding import PositionEmbeddingSine
from torch import nn
from torchvision.models import resnet50
from torch.nn.functional import dropout,linear,softmax
from torchvision.ops.boxes import batched_nms
import ipywidgets as widgets

from tools.visual import filter_boxes, plot_results, AttentionVisualizer
torch.set_grad_enabled(False)
# torch.set_grad_enabled(False)

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
    # device = torch.device(args.device)

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

    # standard PyTorch mean-std input image normalization
    transform_official = T.Compose([
        # T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_local = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # # 使用源代码进行模型加载推理
    # model, criterion, postprocessors = build_model(args)
    # # 加载线上的模型
    # checkpoint = torch.load(args.resume, map_location='cuda')
    # model.load_state_dict(checkpoint['model'])
    # model.to(device)
    # model.eval()


    # 使用hub加载本地模型
    model = torch.hub.load('C:\\Users\\10631\\.cache\\torch\\hub\\facebookresearch_detr_main', 'detr_resnet50', pretrained=True, source='local')
    # model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval()
    # 线上下载图像
    img_path = 'F:/ProjectWorkplace/VOD/Transformer-Series/pics/test.jpg'
    im = Image.open(img_path)
    
    img = transform_local(im).unsqueeze(0)
    # image_tensor=image_tensor.to(device)

    inference_result = model(img)

    probas = inference_result['pred_logits'].softmax(-1)[0, :, :-1].cpu()
    keep = probas.max(-1).values > 0.9
    bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0,keep].cpu(),im.size)
    # scores, boxes = filter_boxes(probas,bboxes_scaled,confidence=0.5, apply_nms=True, iou=0.5)
    # scores = scores.data.numpy()
    # boxes = boxes.data.numpy()
    # bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0, keep], im.size)

    save_path = 'F:/ProjectWorkplace/VOD/Transformer-Series/pics/test_res.jpg'
    # plot_results(im, scores, boxes, CLASSES, COLORS, save_path )
    plot_results(im, probas[keep], bboxes_scaled, CLASSES, COLORS, save_path )

    # prepare for the hook
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    # print(model.backbone)
    outputs = model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    # conv_features = torch.tensor([item.cpu().detach().numpy() for item in conv_features[0]]).cuda()
    # print(conv_features.shape)
    conv_features = conv_features[0]       # [1, 256, 120, 160] [1, 2048, 25, 34] bs num_ch h w
    enc_attn_weights = enc_attn_weights[0] # [1, 300, 300]
    dec_attn_weights = dec_attn_weights[0] # [1, 100, 300]
    # print(conv_features.shape, enc_attn_weights.shape, dec_attn_weights.shape)

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    # print(conv_features)
    # print(conv_features['0'].tensors.shape)
    # [120 ,160]
    
    save_path2 = 'F:/ProjectWorkplace/VOD/Transformer-Series/pics/hot_res.jpg'

    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    colors = COLORS * 100
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        ax = ax_i[0]
        # print(dec_attn_weights[0, idx].shape)
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
    fig.tight_layout()
    fig.show()
    # fig.savefig(save_path2)
    # output of the CNN
    f_map = conv_features['0']
    print("Encoder attention:      ", enc_attn_weights[0].shape)
    print("Feature map:            ", f_map.tensors.shape)
    # get the HxW shape of the feature maps of the CNN
    shape = f_map.tensors.shape[-2:]
    # and reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape(shape + shape)
    print("Reshaped self-attention:", sattn.shape)
    
    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 32

    # let's select 4 reference points for visualization
    idxs = [(200, 200), (280, 400), (200, 600), (440, 800),]

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention{idx_o}')

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(im)
    for (y, x) in idxs:
        scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.axis('off')
    plt.show()
    
    w = AttentionVisualizer(model, transform_official)
    w.run()

    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR test model script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
