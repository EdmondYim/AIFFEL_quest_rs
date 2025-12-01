import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


batch_size = 4


class DiceLoss(nn.Module):
    """
    Dice Loss - Interior Metric과 직접 정렬

    Dice coefficient를 기반으로 한 손실 함수로,
    클래스 불균형 문제에 강건하고 영역(interior) 예측에 효과적
    """

    def __init__(self, smooth=1.0, num_classes=2, ignore_index=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - 모델 출력 (logits)
            target: (B, H, W) - ground truth labels (long type)
        Returns:
            dice_loss: scalar
        """
        # Softmax를 적용하여 확률로 변환
        pred = F.softmax(pred, dim=1)

        # Target을 one-hot encoding으로 변환
        target_one_hot = F.one_hot(
            target, num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(
            0, 3, 1, 2).float()  # (B, C, H, W)

        # ignore_index에 해당하는 픽셀을 마스킹
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float(
            ).unsqueeze(1)  # (B, 1, H, W)
            pred = pred * mask
            target_one_hot = target_one_hot * mask

        # Dice coefficient 계산
        intersection = (pred * target_one_hot).sum(dim=(2, 3))  # (B, C)
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (B, C)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # ignore_index 클래스를 제외하고 평균 계산
        if self.ignore_index is not None:
            # ignore_index 클래스를 제외한 나머지 클래스들의 평균
            valid_classes = [i for i in range(
                self.num_classes) if i != self.ignore_index]
            dice = dice[:, valid_classes].mean()
        else:
            dice = dice.mean()

        # Dice Loss = 1 - Dice coefficient
        dice_loss = 1.0 - dice

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 어려운 샘플에 집중

    클래스 불균형이 심한 경우 유용
    쉬운 샘플의 가중치를 줄이고 어려운 샘플에 집중

    참고: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=1.0, gamma=2.0, num_classes=2, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - 모델 출력 (logits)
            target: (B, H, W) - ground truth labels
        Returns:
            focal_loss: scalar
        """
        # Cross Entropy Loss 계산 (ignore_index 적용)
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(
                # (B, H, W)
                pred, target, ignore_index=self.ignore_index, reduction='none')
        else:
            ce_loss = F.cross_entropy(
                pred, target, reduction='none')  # (B, H, W)

        # Softmax 확률
        p = F.softmax(pred, dim=1)

        # Target에 해당하는 확률 추출
        target_one_hot = F.one_hot(
            target, num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(
            0, 3, 1, 2).float()  # (B, C, H, W)

        p_t = (p * target_one_hot).sum(dim=1)  # (B, H, W)

        # Focal Loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss

    클래스 불균형이 심한 데이터셋에 효과적인 조합.
    Focal Loss는 어려운 샘플에 집중하고, Dice Loss는 전반적인 영역 중첩도를 최적화합니다.
    """

    def __init__(self, num_classes=2, focal_alpha=1.0, focal_gamma=2.0, smooth=1.0, ignore_index=0):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, num_classes=num_classes, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(
            smooth=smooth, num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - 모델 출력 (logits)
            target: (B, H, W) - ground truth labels
        Returns:
            total_loss: scalar
            loss_dict: 각 손실의 값을 담은 딕셔너리
        """
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)

        # 단순 합산 (필요에 따라 가중치 조절 가능)
        total_loss = focal + dice

        loss_dict = {
            'total_loss': total_loss.item(),
            'focal_loss': focal.item(),
            'dice_loss': dice.item()
        }

        return total_loss, loss_dict


# 이전의 BoundaryLoss와 CombinedLoss는 참고용으로 남겨두거나 필요 시 사용
class BoundaryLoss(nn.Module):
    """
    Boundary Loss - Boundary Metric과 직접 정렬
    """

    def __init__(self, num_classes=2, ignore_index=0):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def compute_distance_map(self, target):
        batch_size, height, width = target.shape
        distance_maps = torch.zeros(
            batch_size, self.num_classes, height, width)

        for b in range(batch_size):
            for c in range(self.num_classes):
                if c == self.ignore_index:
                    continue
                mask = (target[b] == c).cpu().numpy().astype(np.uint8)
                if mask.sum() > 0:
                    posmask = mask
                    negmask = 1 - mask
                    pos_dist = distance_transform_edt(posmask)
                    neg_dist = distance_transform_edt(negmask)
                    distance_map = neg_dist - pos_dist
                    distance_maps[b, c] = torch.from_numpy(distance_map)

        return distance_maps.to(target.device)

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        distance_maps = self.compute_distance_map(target)
        boundary_loss = (pred * distance_maps).mean()
        return boundary_loss


if __name__ == '__main__':
    # 테스트 코드
    print("=" * 60)
    print("Focal + Dice Loss 테스트 (ignore_index=0)")
    print("=" * 60)

    num_classes = 2
    height, width = 64, 64

    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    target[0, :10, :10] = 0  # unlabeled

    print(f"\n입력 예측 크기: {pred.shape}")
    print(f"입력 타겟 크기: {target.shape}")

    criterion = FocalDiceLoss(num_classes=num_classes, ignore_index=0)
    total_loss, loss_dict = criterion(pred, target)

    print(f"\nTotal Loss: {loss_dict['total_loss']:.4f}")
    print(f"  - Focal Loss: {loss_dict['focal_loss']:.4f}")
    print(f"  - Dice Loss: {loss_dict['dice_loss']:.4f}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
