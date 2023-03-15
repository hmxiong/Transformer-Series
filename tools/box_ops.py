import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    # 删除最后一个张量维度。
    # 返回沿给定维度的所有切片的元组，已经没有它。
    x_c, y_c ,w,h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    # 得到bbox的四个坐标并添加一个维度
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    # 将bbox的坐标信息转化为长宽以及坐标起始信息
    x0, y0, x1 ,y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
          (x1 - x0), (y1 - y0)]
    return torch.stack(b ,dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2]) # [N,M,2]
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:,:,1]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    # 使用的是同样的方式生成的GIoU计算方式
    assert(boxes1[:,2:] >= boxes1[:,:2]).all()
    assert(boxes2[:,2:] >= boxes2[:,:2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:,None,:2], boxes2[:,:2])
    rb = torch.max(boxes1[:,None,2:], boxes2[:,2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def masks_to_boxes(masks):
    if masks.numel()==0:
        return torch.zeros((0,4), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y,x)

    x_mask = (masks * y.unsqueeze(0))
    
