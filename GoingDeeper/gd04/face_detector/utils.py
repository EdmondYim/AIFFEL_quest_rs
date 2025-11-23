import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _intersect(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, 0] * inter[:, :, 1]

def _jaccard(box_a, box_b):
    inter = _intersect(box_a, box_b)

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter

    return inter / union

def _encode_bbox(matched, boxes, variances=[0.1, 0.2]):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - boxes[:, :2]
    g_cxcy /= (variances[0] * boxes[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / boxes[:, 2:]
    g_wh = torch.clamp(g_wh, min=1e-6)
    g_wh = torch.log(g_wh) / variances[1]

    g_wh = torch.where(torch.isinf(g_wh), torch.zeros_like(g_wh), g_wh)

    return torch.cat([g_cxcy, g_wh], dim=1)

def encode_pt(labels, boxes):
    match_threshold = 0.45

    boxes = boxes.float()
    bbox = labels[:, :4]
    conf = labels[:, -1]

    if bbox.numel() == 0:
        return torch.zeros((boxes.size(0), 5), device=boxes.device)

    # 인풋 이미지가 320x256이므로 center format으로 변환
    # IoU 계산을 위해 corner format으로 변환
    boxes_corner = torch.cat([
        boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
        boxes[:, :2] + boxes[:, 2:] / 2   # xmax, ymax
    ], dim=1)

    overlaps = _jaccard(bbox, boxes_corner)

    best_box_overlap, best_box_idx = overlaps.max(dim=1)
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)

    best_truth_overlap[best_box_idx] = 2.0
    best_truth_idx[best_box_idx] = torch.arange(best_box_idx.size(0), device=best_box_idx.device)

    matches_bbox = bbox[best_truth_idx]
    # 원본 센터 포멧으로 변환
    loc_t = _encode_bbox(matches_bbox, boxes)

    conf_t = conf[best_truth_idx]
    conf_t[best_truth_overlap < match_threshold] = 0

    return torch.cat([loc_t, conf_t.unsqueeze(1)], dim=1)

def hard_negative_mining(loss, class_truth, neg_ratio):
    pos_idx = class_truth > 0
    num_pos = pos_idx.sum(dim=1)
    num_neg = num_pos * neg_ratio

    # loss 복사하고 pos_idx에 -inf로 설정하여 neg_idx를 구함
    loss_for_neg = loss.clone()
    loss_for_neg[pos_idx] = -float('inf')
    
    _, rank = loss_for_neg.sort(dim=1, descending=True)
    
    # 가능한 negative 개수를 clamp
    num_available_neg = (~pos_idx).sum(dim=1)
    num_neg = torch.minimum(num_neg, num_available_neg)
    
    neg_idx = rank < num_neg.unsqueeze(1)

    return pos_idx, neg_idx

# [DEBUG MultiBoxLoss]는 GPT에서 제공받은 코드
class MultiBoxLoss:
    def __init__(self, num_classes, neg_pos_ratio=3.0):
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.debug = False

    def __call__(self, y_true, y_pred):
        loc_pred, class_pred = y_pred[..., :4], y_pred[..., 4:]
        loc_truth, class_truth = y_true[..., :4], y_true[..., 4].long()

        if self.debug:
            print(f"[DEBUG MultiBoxLoss] class_pred shape: {class_pred.shape}, class_truth shape: {class_truth.shape}")
            print(f"[DEBUG MultiBoxLoss] class_pred has NaN: {torch.isnan(class_pred).any()}")
            print(f"[DEBUG MultiBoxLoss] class_truth unique values: {torch.unique(class_truth)}")

        temp_loss = F.cross_entropy(class_pred.view(-1, self.num_classes), class_truth.view(-1), reduction='none')
        temp_loss = temp_loss.view(class_truth.size())
        
        if self.debug:
            print(f"[DEBUG MultiBoxLoss] temp_loss has NaN: {torch.isnan(temp_loss).any()}")
            print(f"[DEBUG MultiBoxLoss] temp_loss min: {temp_loss.min().item():.4f}, max: {temp_loss.max().item():.4f}")
        
        pos_idx, neg_idx = hard_negative_mining(temp_loss, class_truth, self.neg_pos_ratio)

        if self.debug:
            print(f"[DEBUG MultiBoxLoss] num_pos: {pos_idx.sum().item()}, num_neg: {neg_idx.sum().item()}")
            print(f"[DEBUG MultiBoxLoss] num_total_mined: {(pos_idx | neg_idx).sum().item()}")

        loss_class = F.cross_entropy(class_pred[pos_idx | neg_idx], class_truth[pos_idx | neg_idx], reduction='sum')

        # mined samples의 accuracy 계산
        with torch.no_grad():
            preds = class_pred[pos_idx | neg_idx].argmax(dim=1)
            targets = class_truth[pos_idx | neg_idx]
            correct = (preds == targets).float().sum()
            total = (pos_idx | neg_idx).float().sum()
            acc = correct / total if total > 0 else torch.tensor(0.0)

        loss_loc = F.smooth_l1_loss(loc_pred[pos_idx], loc_truth[pos_idx], reduction='sum')

        num_pos = pos_idx.float().sum()

        if self.debug:
            print(f"[DEBUG MultiBoxLoss] loss_class (before norm): {loss_class.item():.4f}")
            print(f"[DEBUG MultiBoxLoss] loss_loc (before norm): {loss_loc.item():.4f}")
            print(f"[DEBUG MultiBoxLoss] num_pos: {num_pos.item()}")

        if num_pos > 0:
            loss_class /= num_pos
            loss_loc /= num_pos
        else:
            if self.debug:
                print(f"[DEBUG MultiBoxLoss] WARNING: num_pos is 0!")
            loss_class = torch.tensor(0.0, requires_grad=True, device=y_pred.device)
            loss_loc = torch.tensor(0.0, requires_grad=True, device=y_pred.device)

        # localization loss를 0.5로 scaling하여 loss를 합리적인 범위로 조정
        loss_loc = loss_loc * 0.5
        loss_class = loss_class * 1.0

        if self.debug:
            print(f"[DEBUG MultiBoxLoss] Final loss_class: {loss_class.item():.4f}, loss_loc: {loss_loc.item():.4f}")

        return loss_loc, loss_class, acc

class PiecewiseConstantWarmUpDecay:
    def __init__(self, boundaries, values, warmup_steps, min_lr):
        if len(boundaries) != len(values) - 1:
            raise ValueError("바운더리의 길이가 값의 길이보다 1 작아야 합니다")

        self.boundaries = boundaries
        self.values = values
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        if step <= self.warmup_steps:
            return self.min_lr + step * (self.values[0] - self.min_lr) / self.warmup_steps

        for i, boundary in enumerate(self.boundaries):
            if step <= boundary:
                return self.values[i]

        return self.values[-1]

def MultiStepWarmUpLR(initial_learning_rate, lr_steps, lr_rate, warmup_steps=0, min_lr=0):
    assert warmup_steps <= lr_steps[0]
    assert min_lr <= initial_learning_rate

    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)

    return PiecewiseConstantWarmUpDecay(boundaries=lr_steps, values=lr_steps_value, warmup_steps=warmup_steps, min_lr=min_lr)
