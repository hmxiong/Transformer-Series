import torch
from tools.misc import (NestedTensor, nested_tensor_from_tensor_list, 
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized, inverse_sigmoid)
from tools import box_ops
import torch.nn.functional as F
# from .detr import sigmoid_focal_loss

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes

def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training, 
                   num_queries, num_classes, hidden_dim, label_enc, model_type):
    """
    进行标签和数据的加噪过程， 为去噪算法做准备


    """
    
    if training:
        print('training')
        # targets 是 List[dict]，代表1個 batch 的標籤，其中每個 dict 是每張圖的標籤
        targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
    else:
        num_patterns = dn_args
    
    if num_patterns == 0:
        num_patterns = 1
    
    indicator0 = torch.zeros([num_queries * num_patterns, 1]).cuda()
    # label_enc 是 nn.Embedding()，其 weight 的 shape 是 (num_classes+1, hidden_dim-1)
    # 第一维之所以是 num_classes+1 是因为以下 tgt 的初始化值是 num_classes，因此要求 embedding 矩阵的第一维必须有 num_classes+1；
    # 而第二维之所以是 hidden_dim-1 是因为要留一个位置给以上的 indicator0
    # 由于去噪任务的 label noise 是在 gt label(0~num_classes-1) 上加噪，
    # 因此这里 tgt 的初始化值是 num_classes，代表 non-object，以区去噪任(dn)务和匹配(matching)任务
    # hidden_dim-1 -> num_queries*num_patterns  hidden_dim-1
    if model_type in ['dab']:
        tgt = label_enc(torch.tensor(num_classes).cuda()).repeat(num_queries * num_patterns, 1)
        tgt = torch.cat([tgt, indicator0],dim=1)
        refpoint_emb = embedweight.repeat(num_patterns, 1)
    elif model_type in ['deformable']:
        tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0]*torch.tensor(0).cuda()
        refpoint_emb = embedweight
    # num_queries 4 -> num_queries*num_patterns 4
    if training:
        # 计算一些相关的索引用于后续计算进行query和gt之间的匹配
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        # 该 batch 里每张图中各 gt 在图片中的 index
        # torch.nonzero() 返回的是张量中值不为0的元素的索引，list 中的每个张量shape是(num_gt_img,1)
        know_idx = [torch.nonzero(t) for t in known]
        # 统计一个batch中的gt的数量
        known_num = [sum(k) for k in known] # 得到的是已知label中所含有非零值的数量
        
        # 对 gt 在整个 batch 中计算索引
        # (num_gts_batch,) 其中都是1
        unmask_bbox = unmask_label = torch.cat(known)
        # gt_labels num_gts_batch 
        labels = torch.cat([t['labels'] for t in targets])
        # gt_boxes num_gts_batch 4
        boxes = torch.cat([t['boxes'] for t in targets])
        # 每张图片的 batch 索引，这个变量用于代表各图片是第几张图片
        # (num_gts_batch,)
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i,t in enumerate(targets)])

        # # 将以上“复制”到所有去噪组
        # (num_gts_batch,4)->(scalar*num_gts_batch,4)
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar, 1)
        # 用于准备在label上进行加噪
        known_labels_expaned = known_labels.clone()
        # 用于准备在bbox上进行加噪
        known_bbox_expand = known_bboxs.clone()

        if label_noise_scale > 0:
            # (scalar*num_gts_batch,) 从分布均匀的地方开始采样
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)
            # 对应论文中进行的flip操作，相当与完成了对标签的加噪处理
            new_label = torch.rand_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        if box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand)
            # 对应的中心点坐标 w/2 h/2
            diff[:,:2] = known_bbox_expand[:, 2:] / 2
            # 对应的宽高 w h
            diff[:,2:] = known_bbox_expand[:, 2:]
            # 对坐标点+偏移量从而保证坐标点保持在内部
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2.0 - 1.0), 
                                            diff).cuda() * box_noise_scale
            # 返回能够在范围之内的数据，如果是范围之外的数据则不会返回
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
        
        m = known_labels_expaned.long().to('cuda')
        # 加噪后的类别对应的标签信息 embedding 向量
        # scalar*num_gts_batch -> scalar*num_gts_batch  hidden_dim-1
        input_label_embed  = label_enc(m)
        # 用于指示去噪任务
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
        # 作为去噪任务的 content quries
        # add dn part indicator
        # scalar*num_gts_batch  hidden_dim
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        # 原 gt boxes 是 [0,1] 归一化的数值，于是这里进行反归一化
        # scalar*num_gts_batch  4
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        single_pad = int(max(known_num))
        ''' padding: 使得该 batch 中每张图都有相同数量的 noised labels & noised boxes '''
        pad_size   = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox  = torch.zeros(pad_size, 4).cuda()
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        input_query_bbox  = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid): # 转换成长整型是什么意思
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
        
        tgt_size = pad_size + num_queries * num_patterns
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0

        attn_mask[pad_size:, :pad_size] = True # 开始准备做attn mask的相关操作
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i+ 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx ': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'pad_size': pad_size
        }
    else:
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None
    
    if model_type in ['dab']:
        input_query_label = input_query_label.transpose(0, 1)
        input_query_bbox = input_query_bbox.transpose(0, 1)

    return input_query_label, input_query_bbox, attn_mask, mask_dict

def dn_post_process(outputs_class, outputs_coord, mask_dict):
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
    return outputs_class, outputs_coord

def prepare_for_loss(mask_dict):
    """
    为计算损失函数做准备
    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice']

    known_indice = mask_dict['known_indice']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel()
    return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt):
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_bbox':torch.as_tensor(0.).to('cuda'),
            'tgt_loss_giou':torch.as_tensor(0.).to('cuda'),
        }
    
    loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

    losses = {}
    losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt

    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(tgt_boxes)
    ))
    losses['tgt_loss_giou'] = loss_giou.sum() / num_tgt

    return losses

def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha, log=True):
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce':torch.as_tensor(0.).to('cuda'),
            'tgt_class_error': torch.as_tensor(0.).to('cuda'),
        }
    
    src_logits, tgt_labels = src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)
    
    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_ce': loss_ce}

    losses['tgt_class_error'] =  100 - accuracy(src_logits_, tgt_labels_)[0]

    return losses

def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):

    losses = {}
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_labels, known_bboxs, output_known_class, output_known_coord, \
        num_tgt = prepare_for_loss(mask_dict)
        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
    
    if aux_num:
        for i in range(aux_num):
            if training and 'output_known_lbs_bboxes' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt, focal_alpha)
                l_dict = {k + f'_{i}':v for k , v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_boxes(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}':v for k,v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}':v for k,v in l_dict.items()}
                losses.update(l_dict)
    
    return losses








