"""
Pose Estimation 평가 메트릭스
- PCKh@0.5: Head-normalized Percentage of Correct Keypoints
- Per-joint Accuracy: 각 관절별 정확도
- Mahalanobis Distance: 공분산 기반 거리 측정
"""

import torch
import numpy as np


# MPII 관절 이름 (16개)
JOINT_NAMES = [
    'R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Pelvis', 'Thorax', 'Upper_Neck', 'Head_Top',
    'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'
]

# Head 관절 인덱스 (Upper_Neck=8, Head_Top=9)
UPPER_NECK_IDX = 8
HEAD_TOP_IDX = 9


def get_max_preds(batch_heatmaps):
    """
    히트맵에서 최대값 좌표와 confidence를 추출합니다.
    
    Args:
        batch_heatmaps: (B, C, H, W) 형태의 히트맵 텐서
        
    Returns:
        preds: (B, C, 2) 형태의 좌표 텐서 [x, y]
        maxvals: (B, C, 1) 형태의 confidence 텐서
    """
    batch_size, num_channels, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.view(batch_size, num_channels, -1)
    idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals = torch.amax(heatmaps_reshaped, 2).view(batch_size, num_channels, 1)
    idx = idx.view(batch_size, num_channels, 1)
    preds = torch.cat((idx % w, idx // w), dim=2).float()
    return preds, maxvals


def compute_head_size(gt_heatmaps):
    """
    GT 히트맵에서 head size를 계산합니다 (Upper_Neck ~ Head_Top 거리).
    
    Args:
        gt_heatmaps: (B, C, H, W) 형태의 GT 히트맵
        
    Returns:
        head_sizes: (B,) 형태의 head size 텐서
    """
    preds, _ = get_max_preds(gt_heatmaps)
    
    # Upper_Neck과 Head_Top 좌표
    upper_neck = preds[:, UPPER_NECK_IDX, :]  # (B, 2)
    head_top = preds[:, HEAD_TOP_IDX, :]      # (B, 2)
    
    # Head size = Euclidean distance
    head_sizes = torch.norm(upper_neck - head_top, dim=1)  # (B,)
    
    # 0인 경우 방지 (최소값 설정)
    head_sizes = torch.clamp(head_sizes, min=1.0)
    
    return head_sizes


def pckh(output, target, threshold=0.5):
    """
    PCKh@threshold 계산
    
    Head size로 정규화된 거리가 threshold 이내이면 correct로 판정합니다.
    
    Args:
        output: (B, C, H, W) 예측 히트맵
        target: (B, C, H, W) GT 히트맵
        threshold: PCKh threshold (기본 0.5)
        
    Returns:
        pckh_score: 전체 PCKh 점수 (0~1)
        per_joint_pckh: 각 관절별 PCKh 점수 (C,)
    """
    with torch.no_grad():
        batch_size, num_channels, h, w = output.shape
        
        # 예측 및 GT 좌표 추출
        preds, _ = get_max_preds(output)  # (B, C, 2)
        gt, _ = get_max_preds(target)     # (B, C, 2)
        
        # Head size 계산
        head_sizes = compute_head_size(target)  # (B,)
        
        # 각 관절별 거리 계산
        dist = torch.norm(preds - gt, dim=2)  # (B, C)
        
        # Head size로 정규화
        normalized_dist = dist / head_sizes.unsqueeze(1)  # (B, C)
        
        # threshold 이내이면 correct
        correct = (normalized_dist <= threshold).float()  # (B, C)
        
        # Per-joint PCKh
        per_joint_pckh = correct.mean(dim=0)  # (C,)
        
        # 전체 PCKh
        pckh_score = correct.mean().item()
        
    return pckh_score, per_joint_pckh


def per_joint_accuracy(output, target, threshold_pixels=3.0):
    """
    각 관절별 정확도 계산 (pixel 기준)
    
    Args:
        output: (B, C, H, W) 예측 히트맵
        target: (B, C, H, W) GT 히트맵
        threshold_pixels: 정확도 판정 픽셀 threshold
        
    Returns:
        joint_acc_dict: 관절별 정확도 딕셔너리
        confusion_matrix: 각 관절별 TP, FP, FN 정보
    """
    with torch.no_grad():
        batch_size, num_channels, h, w = output.shape
        
        preds, pred_conf = get_max_preds(output)  # (B, C, 2), (B, C, 1)
        gt, gt_conf = get_max_preds(target)       # (B, C, 2), (B, C, 1)
        
        # 거리 계산
        dist = torch.norm(preds - gt, dim=2)  # (B, C)
        
        # Correct 판정
        correct = (dist <= threshold_pixels).float()  # (B, C)
        
        # 각 관절별 정확도
        per_joint_acc = correct.mean(dim=0)  # (C,)
        
        # 딕셔너리 형태로 변환
        joint_acc_dict = {}
        for i, name in enumerate(JOINT_NAMES):
            if i < num_channels:
                joint_acc_dict[name] = per_joint_acc[i].item()
        
        # Confusion matrix 정보
        confusion_matrix = {
            'per_joint_accuracy': per_joint_acc.cpu().numpy(),
            'mean_accuracy': correct.mean().item(),
            'per_joint_dist_mean': dist.mean(dim=0).cpu().numpy(),
            'per_joint_dist_std': dist.std(dim=0).cpu().numpy()
        }
        
    return joint_acc_dict, confusion_matrix


def mahalanobis_distance(output, target, regularization=1e-6):
    """
    Mahalanobis 거리 계산
    
    공분산 행렬을 고려하여 예측과 GT 간의 거리를 측정합니다.
    
    Args:
        output: (B, C, H, W) 예측 히트맵
        target: (B, C, H, W) GT 히트맵
        regularization: 공분산 행렬 정규화 값
        
    Returns:
        maha_dist: 평균 Mahalanobis 거리
        per_sample_dist: 각 샘플별 Mahalanobis 거리 (B,)
    """
    with torch.no_grad():
        batch_size, num_channels, h, w = output.shape
        
        preds, _ = get_max_preds(output)  # (B, C, 2)
        gt, _ = get_max_preds(target)     # (B, C, 2)
        
        # 오차 벡터 (B, C, 2)
        diff = preds - gt
        
        # 각 샘플의 오차를 flatten (B, C*2)
        diff_flat = diff.view(batch_size, -1)
        
        # 공분산 행렬 계산 (C*2, C*2)
        # 배치 전체에서 공분산 추정
        diff_centered = diff_flat - diff_flat.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(diff_centered.T, diff_centered) / (batch_size - 1 + 1e-8)
        
        # 정규화 (수치 안정성)
        cov_matrix = cov_matrix + regularization * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        
        # 공분산 행렬의 역행렬
        try:
            cov_inv = torch.linalg.inv(cov_matrix)
        except:
            # 역행렬 계산 실패 시 단위 행렬 사용
            cov_inv = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        
        # Mahalanobis 거리 계산: sqrt(diff.T @ cov_inv @ diff)
        per_sample_dist = []
        for i in range(batch_size):
            d = diff_flat[i:i+1]  # (1, C*2)
            maha = torch.sqrt(torch.mm(torch.mm(d, cov_inv), d.T) + 1e-8)
            per_sample_dist.append(maha.item())
        
        per_sample_dist = torch.tensor(per_sample_dist)
        maha_dist = per_sample_dist.mean().item()
        
    return maha_dist, per_sample_dist


def compute_all_metrics(output, target, pckh_threshold=0.5, pixel_threshold=3.0):
    """
    모든 메트릭을 한 번에 계산합니다.
    
    Args:
        output: (B, C, H, W) 예측 히트맵
        target: (B, C, H, W) GT 히트맵
        pckh_threshold: PCKh threshold
        pixel_threshold: Per-joint accuracy pixel threshold
        
    Returns:
        metrics_dict: 모든 메트릭을 포함한 딕셔너리
    """
    # PCKh
    pckh_score, per_joint_pckh = pckh(output, target, pckh_threshold)
    
    # Per-joint accuracy
    joint_acc_dict, confusion_matrix = per_joint_accuracy(output, target, pixel_threshold)
    
    # Mahalanobis distance
    maha_dist, per_sample_maha = mahalanobis_distance(output, target)
    
    metrics_dict = {
        'pckh': pckh_score,
        'pckh_threshold': pckh_threshold,
        'per_joint_pckh': per_joint_pckh.cpu().numpy() if torch.is_tensor(per_joint_pckh) else per_joint_pckh,
        'mean_accuracy': confusion_matrix['mean_accuracy'],
        'per_joint_accuracy': joint_acc_dict,
        'per_joint_dist_mean': confusion_matrix['per_joint_dist_mean'],
        'per_joint_dist_std': confusion_matrix['per_joint_dist_std'],
        'mahalanobis_distance': maha_dist,
    }
    
    return metrics_dict


def print_metrics_summary(metrics_dict):
    """
    메트릭 요약을 출력합니다.
    """
    print("\n" + "="*60)
    print("                    METRICS SUMMARY")
    print("="*60)
    
    print(f"\n[PCKh@{metrics_dict['pckh_threshold']}]")
    print(f"  Overall: {metrics_dict['pckh']:.4f}")
    
    print(f"\n[Mean Pixel Accuracy]")
    print(f"  Overall: {metrics_dict['mean_accuracy']:.4f}")
    
    print(f"\n[Mahalanobis Distance]")
    print(f"  Mean: {metrics_dict['mahalanobis_distance']:.4f}")
    
    print(f"\n[Per-Joint Accuracy]")
    for joint_name, acc in metrics_dict['per_joint_accuracy'].items():
        print(f"  {joint_name:15s}: {acc:.4f}")
    
    print("\n" + "="*60)


# 테스트 코드 - 실제 모델과 데이터로 메트릭 계산
if __name__ == "__main__":
    import os
    from model import StackedHourglassNetwork, SimpleBaseline
    from dataset import create_dataloader
    
    # 설정
    CHECKPOINT_DIR = 'checkpoints'
    VAL_JSON = 'archive/validation.json'
    IMAGE_DIR = 'archive/mpii_human_pose_v1/images'
    BATCH_SIZE = 32
    NUM_HEATMAP = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validation 데이터 로더
    print("Loading validation data...")
    val_loader = create_dataloader(VAL_JSON, IMAGE_DIR, BATCH_SIZE, NUM_HEATMAP, is_train=False)
    print(f"Validation batches: {len(val_loader)}")
    
    # 모델 로드 함수
    def load_model(model_class, model_name):
        model = model_class(num_heatmap=NUM_HEATMAP)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}-best.pt')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}-latest.pt')
        
        if os.path.exists(checkpoint_path):
            print(f"Loading {model_name} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()
            return model
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
    
    # 모델별 메트릭 계산
    def evaluate_model(model, model_name, val_loader, num_batches=10):
        """모델 평가 (일부 배치만 사용)"""
        if model is None:
            return None
        
        print(f"\nEvaluating {model_name}...")
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for i, (images, heatmaps) in enumerate(val_loader):
                if i >= num_batches:
                    break
                
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                
                outputs = model(images)
                # 마지막 스택의 출력 사용
                output = outputs[-1] if isinstance(outputs, list) else outputs
                
                all_outputs.append(output)
                all_targets.append(heatmaps)
        
        # 배치 합치기
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        print(f"  Evaluated on {all_outputs.shape[0]} samples")
        
        # 메트릭 계산
        metrics = compute_all_metrics(all_outputs, all_targets)
        return metrics
    
    # 모델 로드
    model_hg = load_model(StackedHourglassNetwork, 'hourglass')
    model_sb = load_model(SimpleBaseline, 'simplebaseline')
    
    # 평가 실행
    metrics_hg = evaluate_model(model_hg, 'Hourglass', val_loader)
    metrics_sb = evaluate_model(model_sb, 'SimpleBaseline', val_loader)
    
    # 결과 출력
    print("\n" + "="*70)
    print("                    MODEL COMPARISON")
    print("="*70)
    
    if metrics_hg:
        print("\n[Hourglass]")
        print(f"  PCKh@0.5:           {metrics_hg['pckh']:.4f}")
        print(f"  Mean Accuracy:      {metrics_hg['mean_accuracy']:.4f}")
        print(f"  Mahalanobis Dist:   {metrics_hg['mahalanobis_distance']:.4f}")
    
    if metrics_sb:
        print("\n[SimpleBaseline]")
        print(f"  PCKh@0.5:           {metrics_sb['pckh']:.4f}")
        print(f"  Mean Accuracy:      {metrics_sb['mean_accuracy']:.4f}")
        print(f"  Mahalanobis Dist:   {metrics_sb['mahalanobis_distance']:.4f}")
    
    # Per-joint 비교
    if metrics_hg and metrics_sb:
        print("\n" + "-"*70)
        print("Per-Joint Accuracy Comparison:")
        print("-"*70)
        print(f"{'Joint':<15} {'Hourglass':>12} {'SimpleBaseline':>15} {'Winner':>10}")
        print("-"*70)
        
        for joint in JOINT_NAMES:
            hg_acc = metrics_hg['per_joint_accuracy'].get(joint, 0)
            sb_acc = metrics_sb['per_joint_accuracy'].get(joint, 0)
            winner = "HG" if hg_acc > sb_acc else "SB" if sb_acc > hg_acc else "Tie"
            print(f"{joint:<15} {hg_acc:>12.4f} {sb_acc:>15.4f} {winner:>10}")
        
        print("="*70)

