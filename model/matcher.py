import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from tools.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class:float = 1, cost_bbox:float = 1,
                 cost_giou: float = 1, focal_alpha = 0.25, model_type=None):
        super().__init__() # 用于对序列数据进行相应的匹配
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        
        self.focal_alpha = focal_alpha
        self.model_type = model_type

    @torch.no_grad() # 后续的数据不参加梯度的计算
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        # [2,100,92] -> [200, 92] -> [200, 92]概率
        # out_prob = outputs["pred_logits"].flatten(0,1).softmax(-1)
        out_prob = outputs["pred_logits"].flatten(0,1).sigmoid()
        # [2,100,4] -> [200, 4]   [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0,1)
        
        # concat all boxes and labels
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.model_type in ['base']:
            cost_class = -out_prob[:, tgt_ids]
        else:
            alpha = self.focal_alpha
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # 计算相应的giou损失函数带来的影响，但是我的问题是为什么需要问号
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))
        # Final cost matrix   [100, 3]  bs*100个预测框分别和3个gt框的损失矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou* cost_giou
        C = C.view(bs,num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # 匈牙利算法进行二分图匹配  从100个预测框中挑选出最终的3个预测框 分别和gt计算损失  这个组合的总损失是最小的
        # 0: [3]  5, 35, 63   匹配到的gt个预测框idx
        # 1: [3]  1, 0, 2     对应的gt idx
        indices = [linear_sum_assignment(c[i]) for i,c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i,dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                            focal_alpha=args.focal_alpha, model_type=args.model_type)



