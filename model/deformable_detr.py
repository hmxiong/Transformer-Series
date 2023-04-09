"""
Deformable DETR model.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from tools import box_ops
from tools.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .detr import (dice_loss, sigmoid_focal_loss, MLP, PostProcess, SetCriterion)
# from .detr import (dice_loss, sigmoid_focal_loss, MLP)
from .deformable_transformer import build_deformable_transformer
from .dn_components import prepare_for_dn, dn_post_process
from .transformer import get_clone
import copy

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 use_dab=True, 
                 num_patterns=0, 
                 random_refpoints_xy=False,
                 use_dn=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.ffn_hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        # wether to us DeNoising
        self.use_dn = use_dn
        if self.use_dn:
            self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
            self.num_classes = num_classes


        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                if self.use_dn:
                    self.tgt_embed = nn.Embedding(num_queries, hidden_dim-1) # # for indicator
                else:
                    self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    # import ipdb; ipdb.set_trace()
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
        
        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        # 根据生成的特征图等级来产生相应的骨干网络 3个1*1cov和1个3*3conv用于生成
        # 统一256维的特征token
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            # 使用的是xavire方式进行初始化
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = get_clone(self.class_embed, num_pred)
            self.bbox_embed = get_clone(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor, dn_args=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # pos: 3个不同尺度的特征对应的3个位置编码(这里一步到位直接生成经过1x1conv降维后的位置编码)
        # 0: [bs, 256, H/8, W/8]  1: [bs, 256, H/16, W/16]  2: [bs, 256, H/32, W/32]
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                # C5层输出 bs x 2048 x H/32 x W/32 x  -> bs x 256 x H/64 x W/64     3x3Conv s=2
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                # 因为特征尺寸直接砍一半所以maks的尺寸也要做变化[bs, H/32, H/32] -> [bs, H/64, W/64]
                # 使用的是线性插值法来完成相应的尺寸变换
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # 生成这一层的位置编码  [bs, 256, H/64, W/64]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                
        # srcs:  list4  0=[bs,256,H/8,W/8] 1=[bs,256,H/16,W/16] 2=[bs,256,H/32,W/32] 3=[bs,256,H/64,W/64]
        # masks: list4  0=[bs,H/8,W/8]     1=[bs,H/16,W/16]     2=[bs,H/32,W/32]     3=[bs,H/64,W/64]
        # pos:   list4  0=[bs,256,H/8,W/8] 1=[bs,256,H/16,W/16] 2=[bs,256,H/32,W/32] 3=[bs,256,H/64,W/64]

        # query_embeds = None
        # if not self.two_stage:
        #     query_embeds = self.query_embed.weight
        if self.two_stage:
            query_embeds = None
            attn_mask = None
        elif self.use_dab:
            if self.num_patterns == 0:
                refanchor = self.refpoint_embed.weight      # nq, 4
                if self.use_dn == False:
                    tgt_embed = self.tgt_embed.weight           # nq, 256
                    query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
                    attn_mask = None
                elif self.use_dn:
                    tgt_all_embed = tgt_embed = self.tgt_embed.weight
                    # prepare for dn
                    input_query_label, input_query_bbox, attn_mask, mask_dict = \
                        prepare_for_dn(dn_args, tgt_all_embed, refanchor, src.size(0), self.training, self.num_queries, self.num_classes,
                                    self.hidden_dim, self.label_enc, model_type='deformable')
                    query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2)           # nq, 256
            else:
                # multi patterns
                tgt_embed = self.tgt_embed.weight           # nq, 256
                pat_embed = self.patterns_embed.weight      # num_pat, 256
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1) # nq*num_pat, 256
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) # nq*num_pat, 256
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      # nq*num_pat, 4
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
                attn_mask = None
        else:
            query_embeds = self.query_embed.weight
            attn_mask = None
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embeds, attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # [bs, 300, 2] -> [bs, 300, 2]  反归一化   因为reference在定义的时候就sigmoid归一化了
            reference = inverse_sigmoid(reference)
            # 分类头 1个全连接层 [bs, 300, 256] -> [bs, 300, num_classes]
            outputs_class = self.class_embed[lvl](hs[lvl])
            # 回归头 3个全连接层 [bs, 300, 256] -> [bs, 300, 4]  xywh  xy是偏移量
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference # 偏移量 + 参考点坐标 -> 最终xy坐标
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if self.use_dn:
            # dn post process
            outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        # 'pred_logits': 最后一层的分类头输出 [bs, 300, num_classes]
        # 'pred_boxes': 最后一层的回归头输出 [bs, 300, xywh(归一化)]
        # 'aux_outputs': 其他中间5层的分类头和输出头
        if self.use_dn:
            return out, mask_dict
        else:
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build_deformable_detr(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)
    # print(backbone.num_channels)
    model_type = args.model_type
    transformer = build_deformable_transformer(args)
    print("model type: deformable")
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_dab=True,
        num_patterns=args.num_patterns,
        random_refpoints_xy=args.random_refpoints_xy,
        use_dn = args.use_dn
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # dn loss
    if args.use_dn:
        weight_dict['tgt_loss_ce'] = args.cls_loss_coef
        weight_dict['tgt_loss_bbox'] = args.bbox_loss_coef
        weight_dict['tgt_loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=args.eos_coef, focal_alpha=args.focal_alpha,losses=losses, model_type=model_type)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(model_type, num_select=args.num_select)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
