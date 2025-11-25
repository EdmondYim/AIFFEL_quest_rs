"""
RetinaNet 모델 구현
KITTI 객체 탐지를 위한 RetinaNet 모델과 Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import nms
from typing import List, Tuple
import numpy as np


# ============================================================================
# Anchor Box Generation
# ============================================================================

class AnchorBox:
    """
    RetinaNet을 위한 Anchor Box 생성기
    
    Args:
        aspect_ratios: Anchor의 가로세로 비율 리스트
        scales: Anchor 크기 스케일 리스트
        areas: 각 피라미드 레벨별 기본 영역
    """
    
    def __init__(
        self,
        aspect_ratios=[0.5, 1.0, 2.0],
        scales=[2 ** 0, 2 ** (1/3), 2 ** (2/3)],
        areas=[32.0**2, 64.0**2, 128.0**2, 256.0**2, 512.0**2]
    ):
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.areas = areas
        
        self._num_anchors = len(aspect_ratios) * len(scales)
        self._strides = [2 ** i for i in range(3, 8)]  # [8, 16, 32, 64, 128]
        self._anchor_dims = self._compute_dims()
    
    def _compute_dims(self):
        """각 피라미드 레벨과 aspect ratio/scale 조합별 anchor 크기 계산"""
        anchor_dims_all = []
        
        for area in self.areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = torch.sqrt(torch.tensor(area / ratio))
                anchor_width = area / anchor_height
                
                dims = torch.stack([anchor_width, anchor_height], dim=-1)
                dims = dims.unsqueeze(0).unsqueeze(0)
                
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            
            anchor_dims_all.append(torch.stack(anchor_dims, dim=-2))
        
        return anchor_dims_all
    
    def _get_anchors(self, feature_height, feature_width, level):
        """특정 피라미드 레벨의 anchor boxes 생성"""
        # 그리드 중심점 생성
        rx = torch.arange(feature_width, dtype=torch.float32) + 0.5
        ry = torch.arange(feature_height, dtype=torch.float32) + 0.5
        
        centers = torch.stack(torch.meshgrid(rx, ry, indexing='xy'), dim=-1)
        centers = centers * self._strides[level - 3]
        centers = centers.unsqueeze(-2)
        centers = centers.repeat(1, 1, self._num_anchors, 1)
        
        # Anchor 크기 정보
        dims = self._anchor_dims[level - 3].repeat(feature_height, feature_width, 1, 1)
        
        # [center_x, center_y, width, height] 형식
        anchors = torch.cat([centers, dims], dim=-1)
        return anchors.view(feature_height * feature_width * self._num_anchors, 4)
    
    def get_anchors(self, image_height, image_width):
        """이미지 크기에 대한 모든 피라미드 레벨의 anchors 생성"""
        anchors = [
            self._get_anchors(
                torch.ceil(torch.tensor(image_height / (2 ** i))).int().item(),
                torch.ceil(torch.tensor(image_width / (2 ** i))).int().item(),
                i
            )
            for i in range(3, 8)
        ]
        return torch.cat(anchors, dim=0)


# ============================================================================
# Feature Pyramid Network (FPN)
# ============================================================================

class FeaturePyramid(nn.Module):
    """
    Feature Pyramid Network
    ResNet 백본에서 추출한 feature maps를 피라미드 구조로 변환
    """
    
    def __init__(self, backbone):
        super(FeaturePyramid, self).__init__()
        self.backbone = backbone
        
        # 1x1 convolution for lateral connections
        self.conv_c3_1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv_c4_1x1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv_c5_1x1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        
        # 3x3 convolution for smoothing
        self.conv_c3_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_c4_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_c5_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Extra layers for P6, P7
        self.conv_c6_3x3 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv_c7_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        
        self.upsample_2x = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) 입력 이미지
        
        Returns:
            p3, p4, p5, p6, p7: 5개 레벨의 feature maps
        """
        # ResNet backbone feature extraction
        c3_output, c4_output, c5_output = self.backbone(images)
        
        # Top-down pathway
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p3_output = self.conv_c3_1x1(c3_output)
        
        # Add upsampled features (크기를 맞춰서 더하기)
        p5_upsampled = F.interpolate(p5_output, size=p4_output.shape[-2:], mode='nearest')
        p4_output = p4_output + p5_upsampled
        
        p4_upsampled = F.interpolate(p4_output, size=p3_output.shape[-2:], mode='nearest')
        p3_output = p3_output + p4_upsampled
        
        # Apply 3x3 smoothing
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        
        # Extra coarse levels
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(F.relu(p6_output))
        
        return p3_output, p4_output, p5_output, p6_output, p7_output


# ============================================================================
# RetinaNet Heads
# ============================================================================

def build_head(output_filters, bias_init, num_convs=4):
    """
    분류 또는 박스 회귀 헤드 구축
    
    Args:
        output_filters: 출력 채널 수
        bias_init: 마지막 레이어의 bias 초기화 값
        num_convs: convolution 레이어 개수
    """
    layers = []
    
    for _ in range(num_convs):
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
    
    # 최종 예측 레이어
    layers.append(nn.Conv2d(256, output_filters, kernel_size=3, stride=1, padding=1))
    
    # Weight 초기화
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    
    # 마지막 레이어 bias 초기화
    if isinstance(bias_init, (int, float)):
        nn.init.constant_(layers[-1].bias, bias_init)
    
    return nn.Sequential(*layers)


# ============================================================================
# ResNet Backbone
# ============================================================================

def get_resnet50_backbone(pretrained=True):
    """
    ResNet50 백본 생성 (C3, C4, C5 feature maps 반환)
    
    Returns:
        ResNet50Backbone 모듈
    """
    resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
    
    class ResNet50Backbone(nn.Module):
        def __init__(self, resnet_model):
            super(ResNet50Backbone, self).__init__()
            self.conv1 = resnet_model.conv1
            self.bn1 = resnet_model.bn1
            self.relu = resnet_model.relu
            self.maxpool = resnet_model.maxpool
            
            self.layer1 = resnet_model.layer1  # C2
            self.layer2 = resnet_model.layer2  # C3 (256 channels)
            self.layer3 = resnet_model.layer3  # C4 (512 channels)
            self.layer4 = resnet_model.layer4  # C5 (1024 channels)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            c3 = self.layer2(x)  # 1/8
            c4 = self.layer3(c3)  # 1/16
            c5 = self.layer4(c4)  # 1/32
            
            return c3, c4, c5
    
    return ResNet50Backbone(resnet)


# ============================================================================
# RetinaNet Model
# ============================================================================

class RetinaNet(nn.Module):
    """
    RetinaNet 객체 탐지 모델
    
    Args:
        num_classes: 객체 클래스 개수 (배경 제외)
        backbone: Feature 추출을 위한 백본 네트워크
    """
    
    def __init__(self, num_classes, backbone=None):
        super(RetinaNet, self).__init__()
        
        if backbone is None:
            backbone = get_resnet50_backbone(pretrained=True)
        
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes
        
        # Prior probability for rare class
        prior_probability = 0.01
        bias_value = -np.log((1 - prior_probability) / prior_probability)
        
        # 9 = 3 aspect ratios * 3 scales
        self.cls_head = build_head(9 * num_classes, bias_value)
        self.box_head = build_head(9 * 4, 0.0)
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) 입력 이미지
        
        Returns:
            predictions: (B, num_anchors, 4 + num_classes)
                - 처음 4개: bbox 회귀 예측
                - 나머지: 클래스 로짓
        """
        features = self.fpn(images)
        N = images.size(0)
        
        cls_outputs = []
        box_outputs = []
        
        for feature in features:
            box_output = self.box_head(feature)
            cls_output = self.cls_head(feature)
            
            # Reshape: (B, C, H, W) -> (B, H*W*9, 4 or num_classes)
            box_output = box_output.permute(0, 2, 3, 1).contiguous()
            box_output = box_output.view(N, -1, 4)
            
            cls_output = cls_output.permute(0, 2, 3, 1).contiguous()
            cls_output = cls_output.view(N, -1, self.num_classes)
            
            box_outputs.append(box_output)
            cls_outputs.append(cls_output)
        
        cls_outputs = torch.cat(cls_outputs, dim=1)
        box_outputs = torch.cat(box_outputs, dim=1)
        
        return torch.cat([box_outputs, cls_outputs], dim=-1)


# ============================================================================
# Loss Functions
# ============================================================================

class RetinaNetBoxLoss(nn.Module):
    """Smooth L1 Loss for bounding box regression"""
    
    def __init__(self, delta=1.0):
        super(RetinaNetBoxLoss, self).__init__()
        self._delta = delta
    
    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: (B, num_anchors, 4) ground truth boxes
            y_pred: (B, num_anchors, 4) predicted boxes
        
        Returns:
            loss: (B, num_anchors) box regression loss
        """
        difference = y_true - y_pred
        absolute_difference = torch.abs(difference)
        squared_difference = difference ** 2
        
        loss = torch.where(
            absolute_difference < self._delta,
            0.5 * squared_difference,
            absolute_difference - 0.5
        )
        
        return torch.sum(loss, dim=-1)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: 클래스 가중치 (일반적으로 0.25)
        gamma: focusing parameter (일반적으로 2.0)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
    
    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: (B, num_anchors, num_classes) one-hot encoded labels
            y_pred: (B, num_anchors, num_classes) predicted logits
        
        Returns:
            loss: (B, num_anchors) focal loss
        """
        cross_entropy = F.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction='none'
        )
        
        probs = torch.sigmoid(y_pred)
        alpha = torch.where(y_true == 1.0, self._alpha, 1.0 - self._alpha)
        pt = torch.where(y_true == 1.0, probs, 1.0 - probs)
        
        loss = alpha * torch.pow(1.0 - pt, self._gamma) * cross_entropy
        
        return torch.sum(loss, dim=-1)


class RetinaNetLoss(nn.Module):
    """
    RetinaNet 전체 손실 함수 (Focal Loss + Smooth L1 Loss)
    
    Args:
        num_classes: 객체 클래스 개수
        alpha: Focal loss alpha
        gamma: Focal loss gamma
        delta: Smooth L1 delta
    """
    
    def __init__(self, num_classes=8, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__()
        self._clf_loss = FocalLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes
    
    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: (B, num_anchors, 5) 
                - [:, :, :4]: bbox targets
                - [:, :, 4]: class labels (-2: ignore, -1: negative, 0+: positive)
            y_pred: (B, num_anchors, 4 + num_classes) model predictions
        
        Returns:
            loss: scalar, 배치 전체의 평균 손실
        """
        batch_size = y_true.size(0)
        num_anchors = y_true.size(1)
        
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_predictions = y_pred[:, :, 4:]
        
        # Class labels 처리: -1(배경)은 [0,0,...,0], -2(무시)는 별도 처리, 0+는 정상 클래스
        raw_cls_labels = y_true[:, :, 4].long()
        
        # One-hot encoding (배경 -1도 올바르게 처리)
        cls_labels = torch.zeros(batch_size, num_anchors, self._num_classes, 
                                 device=y_true.device, dtype=torch.float32)
        
        # Positive anchors만 one-hot 설정
        positive_mask = (raw_cls_labels >= 0)
        if positive_mask.any():
            cls_labels[positive_mask] = F.one_hot(
                raw_cls_labels[positive_mask], 
                num_classes=self._num_classes
            ).float()
        
        # Masks
        positive_mask_float = (raw_cls_labels >= 0).float()
        ignore_mask = (raw_cls_labels == -2).float()
        
        # Classification loss (ignore mask 적용)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        clf_loss = torch.where(ignore_mask == 1.0, torch.zeros_like(clf_loss), clf_loss)
        
        # Box regression loss (positive anchors만)
        box_loss = self._box_loss(box_labels, box_predictions)
        box_loss = torch.where(positive_mask_float == 1.0, box_loss, torch.zeros_like(box_loss))
        
        # 배치 전체 num_pos로 정규화 (안정성 향상)
        total_num_pos = torch.sum(positive_mask_float)
        total_num_pos = torch.clamp(total_num_pos, min=1.0)
        
        # 전체 합산 후 정규화
        total_clf_loss = torch.sum(clf_loss) / total_num_pos
        total_box_loss = torch.sum(box_loss) / total_num_pos
        
        total_loss = total_clf_loss + total_box_loss
        
        # Scalar 반환 (train.py에서 .mean() 불필요)
        return total_loss


# ============================================================================
# Utility Functions
# ============================================================================

def convert_to_corners(boxes):
    """중심점 형식을 corner 형식으로 변환"""
    return torch.cat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0,
         boxes[..., :2] + boxes[..., 2:] / 2.0],
        dim=-1
    )


def convert_to_xywh(boxes):
    """Corner 형식을 중심점 형식으로 변환"""
    return torch.cat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0,
         boxes[..., 2:] - boxes[..., :2]],
        dim=-1
    )


class DecodePredictions(nn.Module):
    """
    RetinaNet 예측을 실제 bounding boxes로 디코딩
    
    Args:
        num_classes: 클래스 개수
        confidence_threshold: 신뢰도 임계값
        nms_iou_threshold: NMS IoU 임계값
        max_detections: 최대 탐지 개수
        box_variance: Anchor box 인코딩 시 사용된 분산
    """
    
    def __init__(
        self,
        num_classes=8,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2]
    ):
        super(DecodePredictions, self).__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self._anchor_box = AnchorBox()
        self._box_variance = torch.tensor(box_variance, dtype=torch.float32)
    
    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        """Anchor boxes와 예측값으로부터 실제 boxes 디코딩"""
        boxes = box_predictions * self._box_variance.to(box_predictions.device)
        
        boxes = torch.cat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                torch.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:]
            ],
            dim=-1
        )
        
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed
    
    def forward(self, images, predictions):
        """
        Args:
            images: (B, 3, H, W) 입력 이미지
            predictions: (B, num_anchors, 4 + num_classes) 모델 예측
        
        Returns:
            List of tuples (boxes, scores, classes) for each image in batch
        """
        image_shape = images.shape
        anchor_boxes = self._anchor_box.get_anchors(image_shape[2], image_shape[3])
        anchor_boxes = anchor_boxes.to(predictions.device)
        
        box_predictions = predictions[:, :, :4]
        cls_predictions = torch.sigmoid(predictions[:, :, 4:])
        
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        
        batch_results = []
        
        for i in range(boxes.shape[0]):
            selected_boxes = []
            selected_scores = []
            selected_classes = []
            
            for class_id in range(self.num_classes):
                class_scores = cls_predictions[i, :, class_id]
                mask = class_scores > self.confidence_threshold
                
                if mask.sum() == 0:
                    continue
                
                class_boxes = boxes[i, mask]
                class_scores_filtered = class_scores[mask]
                
                # NMS
                keep = nms(class_boxes, class_scores_filtered, self.nms_iou_threshold)
                
                selected_boxes.append(class_boxes[keep])
                selected_scores.append(class_scores_filtered[keep])
                selected_classes.append(torch.full((len(keep),), class_id, dtype=torch.int64))
            
            if len(selected_boxes) > 0:
                all_boxes = torch.cat(selected_boxes, dim=0)
                all_scores = torch.cat(selected_scores, dim=0)
                all_classes = torch.cat(selected_classes, dim=0)
                
                # 최대 탐지 개수 제한
                if len(all_boxes) > self.max_detections:
                    top_scores, top_indices = torch.topk(all_scores, self.max_detections)
                    all_boxes = all_boxes[top_indices]
                    all_scores = top_scores
                    all_classes = all_classes[top_indices]
            else:
                all_boxes = torch.empty((0, 4), device=boxes.device)
                all_scores = torch.empty((0,), device=boxes.device)
                all_classes = torch.empty((0,), dtype=torch.int64, device=boxes.device)
            
            batch_results.append((all_boxes, all_scores, all_classes))
        
        return batch_results


# 테스트 코드

def test_model():
    """모델 기본 동작 테스트"""
    print("=" * 60)
    print("RetinaNet 모델 테스트")
    print("=" * 60)
    
    # 모델 생성
    num_classes = 8
    model = RetinaNet(num_classes=num_classes)
    model.eval()
    
    # 더미 입력
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 800, 800)
    
    print(f"\n입력 shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(dummy_input)
    
    print(f"출력 shape: {predictions.shape}")
    print(f"  - Batch size: {predictions.shape[0]}")
    print(f"  - Num anchors: {predictions.shape[1]}")
    print(f"  - Predictions per anchor: {predictions.shape[2]} (4 bbox + {num_classes} classes)")
    
    # Loss 테스트
    print("\nLoss 함수 테스트...")
    loss_fn = RetinaNetLoss(num_classes=num_classes)
    
    # 더미 타겟 생성
    dummy_target = torch.randn(batch_size, predictions.shape[1], 5)
    dummy_target[:, :, 4] = torch.randint(-2, num_classes, (batch_size, predictions.shape[1])).float()
    
    loss = loss_fn(dummy_target, predictions)
    print(f"Loss shape: {loss.shape}")
    print(f"Loss 값: {loss}")
    
    # Decode 테스트
    print("\nDecode 테스트...")
    decoder = DecodePredictions(num_classes=num_classes, confidence_threshold=0.5)
    
    with torch.no_grad():
        decoded = decoder(dummy_input, predictions)
    
    print(f"Decoded results: {len(decoded)} images")
    for i, (boxes, scores, classes) in enumerate(decoded):
        print(f"  Image {i}: {len(boxes)} detections")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == '__main__':
    test_model()
