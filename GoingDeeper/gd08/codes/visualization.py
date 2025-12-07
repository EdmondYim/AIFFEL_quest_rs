import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import os
from PIL import Image
from torchvision import transforms
from utils import calculate_crop_box

# --- Constants ---
R_ANKLE = 0
R_KNEE = 1
R_HIP = 2
L_HIP = 3
L_KNEE = 4
L_ANKLE = 5
PELVIS = 6
THORAX = 7
UPPER_NECK = 8
HEAD_TOP = 9
R_WRIST = 10
R_ELBOW = 11
R_SHOULDER = 12
L_SHOULDER = 13
L_ELBOW = 14
L_WRIST = 15

MPII_BONES = [
    [R_ANKLE, R_KNEE],
    [R_KNEE, R_HIP],
    [R_HIP, PELVIS],
    [L_HIP, PELVIS],
    [L_HIP, L_KNEE],
    [L_KNEE, L_ANKLE],
    [PELVIS, THORAX],
    [THORAX, UPPER_NECK],
    [UPPER_NECK, HEAD_TOP],
    [R_WRIST, R_ELBOW],
    [R_ELBOW, R_SHOULDER],
    [THORAX, R_SHOULDER],
    [THORAX, L_SHOULDER],
    [L_SHOULDER, L_ELBOW],
    [L_ELBOW, L_WRIST]
]

# --- Functions ---

def visualize_training_comparison(history1, history2, name1, name2, epoch_start=1, y_lim=True):
    """
    두 모델의 학습 기록(history)을 받아
    2x2 그리드로 시각화 (Row=Accuracy, Column=Model)
    epoch_start: 특정 epoch 이후부터만 시각화
    """

    import matplotlib.pyplot as plt

    # epoch_start 적용 (1부터 시작한다고 가정)
    idx1 = epoch_start - 1
    epochs1 = range(epoch_start, len(history1['train_loss']) + 1)
    epochs2 = range(epoch_start, len(history2['train_loss']) + 1)


    # slice 적용
    h1_train_loss = history1['train_loss'][idx1:]
    h1_val_loss   = history1['val_loss'][idx1:]
    h1_train_acc  = history1['train_acc'][idx1:]
    h1_val_acc    = history1['val_acc'][idx1:]
    
    h2_train_loss = history2['train_loss'][idx1:]
    h2_val_loss   = history2['val_loss'][idx1:]
    h2_train_acc  = history2['train_acc'][idx1:]
    h2_val_acc    = history2['val_acc'][idx1:]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # -----------------------
    # Y축 범위 계산
    # -----------------------
    if y_lim:
        all_losses = h1_train_loss + h1_val_loss + h2_train_loss + h2_val_loss
        loss_min, loss_max = min(all_losses), max(all_losses)
    
        all_accs = h1_train_acc + h1_val_acc + h2_train_acc + h2_val_acc
        acc_min, acc_max = min(all_accs), max(all_accs)

    # --- Column 1: Loss ---
    axes[0, 0].plot(epochs1, h1_train_loss, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs1, h1_val_loss,   'r-', label='Val Loss')
    axes[0, 0].set_title(f'{name1} - Loss (epoch ≥ {epoch_start})')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    if y_lim:
        axes[0, 0].set_ylim(loss_min, loss_max)
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs2, h2_train_loss, 'b-', label='Train Loss')
    axes[0, 1].plot(epochs2, h2_val_loss,   'r-', label='Val Loss')
    axes[0, 1].set_title(f'{name2} - Loss (epoch ≥ {epoch_start})')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    if y_lim:
        axes[0, 1].set_ylim(loss_min, loss_max)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # --- Column 2: Accuracy ---
    axes[1, 0].plot(epochs1, h1_train_acc, 'b-', label='Train Acc')
    axes[1, 0].plot(epochs1, h1_val_acc,   'r-', label='Val Acc')
    axes[1, 0].set_title(f'{name1} - Accuracy (PCKh)')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Accuracy')
    if y_lim:
        axes[1, 0].set_ylim(acc_min, acc_max)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs2, h2_train_acc, 'b-', label='Train Acc')
    axes[1, 1].plot(epochs2, h2_val_acc,   'r-', label='Val Acc')
    axes[1, 1].set_title(f'{name2} - Accuracy (PCKh)')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy')
    if y_lim:
        axes[1, 1].set_ylim(acc_min, acc_max)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def find_max_coordinates(heatmaps):
    # heatmaps: (H, W, C)
    H, W, C = heatmaps.shape
    # reshape to (H*W, C)
    flatten_heatmaps = heatmaps.reshape(-1, C)
    # 각 채널 별 최대값 인덱스 (flattened index)
    indices = torch.argmax(flatten_heatmaps, dim=0)
    # y 좌표: index // H, x 좌표: index - H * y
    y = indices // H
    x = indices - H * y
    # 반환: (C, 2) 텐서, 각 행이 [x, y] 좌표
    return torch.stack([x, y], dim=1)

def extract_keypoints_from_heatmap(heatmaps, confidence_threshold=0.3):
    """
    heatmaps: (H, W, C) 텐서 (예: (64,64,16))
    confidence_threshold: 히트맵 최댓값이 이 값 이하면 invalid로 처리
    """
    H, W, C = heatmaps.shape
    max_keypoints = find_max_coordinates(heatmaps)  # shape: (C, 2) with [x, y] per channel

    # 각 채널별 최댓값 계산 (confidence)
    heatmaps_permuted = heatmaps.permute(2, 0, 1)  # (C, H, W)
    max_values = heatmaps_permuted.view(C, -1).max(dim=1)[0]  # (C,)

    # pad heatmaps
    padded = F.pad(heatmaps_permuted, (1, 1, 1, 1))  # pad (left, right, top, bottom)
    padded_heatmaps = padded.permute(1, 2, 0)  # (H+2, W+2, C)

    adjusted_keypoints = []
    for i, keypoint in enumerate(max_keypoints):
        # Confidence check: 히트맵 최댓값이 threshold 이하면 invalid
        if max_values[i].item() < confidence_threshold:
            adjusted_keypoints.append((-1.0, -1.0))  # Invalid marker
            continue

        # 기존 keypoint의 좌표에 패딩 오프셋 추가
        max_x = int(keypoint[0].item()) + 1
        max_y = int(keypoint[1].item()) + 1

        # 3x3 패치를 추출 (채널 i)
        patch = padded_heatmaps[max_y-1:max_y+2, max_x-1:max_x+2, i]  # (3,3)
        # 중앙 값 제거
        patch[1, 1] = 0
        # 패치 내 최대값의 index를 찾음
        flat_patch = patch.reshape(-1)
        index = torch.argmax(flat_patch).item()

        next_y = index // 3
        next_x = index % 3
        delta_y = (next_y - 1) / 4.0
        delta_x = (next_x - 1) / 4.0

        adjusted_x = keypoint[0].item() + delta_x
        adjusted_y = keypoint[1].item() + delta_y
        adjusted_keypoints.append((adjusted_x, adjusted_y))

    # 리스트를 텐서로 변환
    adjusted_keypoints = torch.tensor(adjusted_keypoints)
    # Invalid keypoints (-1, -1)는 그대로 유지, valid만 정규화
    # 먼저 mask 생성
    valid_mask = adjusted_keypoints[:, 0] >= 0
    normalized_keypoints = adjusted_keypoints.clone()
    # Valid keypoints만 clamp & normalize
    normalized_keypoints[valid_mask] = torch.clamp(adjusted_keypoints[valid_mask], 0, H) / H
    
    return normalized_keypoints

def crop_image_using_keypoints(image, keypoints, scale, margin=0.2):
    """
    keypoints와 scale 정보를 사용하여 학습 때와 동일하게 이미지를 crop합니다.
    keypoints: list of [x, y]
    """
    img_width, img_height = image.size
    
    # 유효한 keypoint (값 > 0)만 선택
    # MPII 데이터셋에서 좌표가 0보다 작으면 invalid로 간주할 수도 있지만,
    # 여기서는 dataset.py의 로직(>0)을 따릅니다.
    kx = [p[0] for p in keypoints if p[0] > 0]
    ky = [p[1] for p in keypoints if p[1] > 0]
    
    # calculate_crop_box는 tensor나 list를 받습니다.
    effective_xmin, effective_ymin, effective_xmax, effective_ymax, _, _, _, _ = calculate_crop_box(
        kx, ky, scale, img_width, img_height, margin
    )
    
    cropped_image = image.crop((effective_xmin, effective_ymin, effective_xmax, effective_ymax))
    
    return cropped_image

def predict(model, image_path, keypoints=None, scale=None):
    # 이미지 로드 (RGB 모드)
    image = Image.open(image_path).convert("RGB")

    # Keypoints/Scale이 제공되면 Crop 수행 (학습과 동일한 로직)
    if keypoints is not None and scale is not None:
        image = crop_image_using_keypoints(image, keypoints, scale)

    # 전처리: 리사이즈, 텐서 변환, [-1, 1] 범위로 스케일링
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),         # 결과: [0, 1]
        transforms.Lambda(lambda x: x * 2 - 1)  # [0,1] -> [-1,1]
    ])
    inputs = preprocess(image)          # shape: (C, H, W)
    inputs = inputs.unsqueeze(0)        # 배치 차원 추가: (1, C, H, W)

    # 모델의 device에 맞게 입력을 이동시킵니다.
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)         # outputs가 리스트가 아닐 수 있음

    if not isinstance(outputs, list):
        outputs = [outputs]

    # 마지막 스택의 결과를 사용한다고 가정 (출력 shape: (1, num_heatmap, H, W))
    heatmap_tensor = outputs[-1].squeeze(0)      # (num_heatmap, H, W)
    # extract_keypoints_from_heatmap 함수는 (H, W, num_heatmap) 형태를 기대하므로 차원 순서를 변경
    heatmap_tensor = heatmap_tensor.permute(1, 2, 0)  # (H, W, num_heatmap)

    # detach, CPU로 이동 후 extract_keypoints_from_heatmap에 전달
    heatmap = heatmap_tensor.detach().cpu()
    kp = extract_keypoints_from_heatmap(heatmap)

    return image, kp

def to_numpy_image(image):
    if torch.is_tensor(image):
        image_np = image.detach().cpu().numpy()
        # 이미지 채널이 첫 번째 차원이라면, (C, H, W) -> (H, W, C)
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
    elif not isinstance(image, np.ndarray):
        # PIL.Image인 경우
        image_np = np.array(image)
    else:
        image_np = image

    # 이미지가 [-1, 1] 범위라면 [0, 1]로 변환
    if image_np.min() < 0:
        image_np = (image_np + 1) / 2.0
    return image_np

def draw_keypoints_on_image(image, keypoints, index=None, accuracy=None, ax=None):
    image_np = to_numpy_image(image)

    if ax is None:
        fig, ax = plt.subplots(1)

    ax.imshow(image_np)
    for i, joint in enumerate(keypoints):
        # Skip invalid keypoints (marked as -1)
        if joint[0] < 0 or joint[1] < 0:
            continue
        joint_x = joint[0] * image_np.shape[1]
        joint_y = joint[1] * image_np.shape[0]
        if index is not None and index != i:
            continue
        ax.scatter(joint_x, joint_y, s=20, c='red', marker='o')

    title = "Keypoints"
    if accuracy is not None:
        if isinstance(accuracy, str):
             title += f" ({accuracy})"
        else:
             title += f" (Acc: {accuracy:.2f})"
    ax.set_title(title)
    ax.axis('off')

def draw_skeleton_on_image(image, keypoints, index=None, ax=None):
    image_np = to_numpy_image(image)

    if ax is None:
        fig, ax = plt.subplots(1)

    ax.imshow(image_np)
    joints = []
    for i, joint in enumerate(keypoints):
        # Keep (-1, -1) for invalid keypoints to maintain indexing
        if joint[0] < 0 or joint[1] < 0:
            joints.append(None)  # Invalid marker
        else:
            joint_x = joint[0] * image_np.shape[1]
            joint_y = joint[1] * image_np.shape[0]
            joints.append((joint_x, joint_y))

    for bone in MPII_BONES:
        if bone[0] < len(joints) and bone[1] < len(joints):
            joint_1 = joints[bone[0]]
            joint_2 = joints[bone[1]]
            # Skip if either joint is invalid
            if joint_1 is None or joint_2 is None:
                continue
            ax.plot([joint_1[0], joint_2[0]], [joint_1[1], joint_2[1]], linewidth=3, alpha=0.7, c='cyan')

    ax.set_title("Skeleton")
    ax.axis('off')

def compare_model_predictions(val_dataset, model_hg, model_sb):
    """
    검증 데이터셋에서 랜덤한 샘플을 하나 뽑아 StackedHourglass와 SimpleBaseline의 예측 결과를 비교 시각화합니다.
    """
    # 1. 검증 데이터셋에서 샘플 하나 가져오기
    idx = random.randint(0, len(val_dataset) - 1)
    sample_image, sample_anno = val_dataset[idx]

    # Ground Truth Keypoints 정규화 (0~1)
    img_w, img_h = sample_image.size
    gt_joints = torch.tensor(sample_anno['joints']) / torch.tensor([img_w, img_h])
    
    # Keypoints/Scale 정보 추출 (학습과 동일한 Crop을 위해)
    joints = sample_anno['joints']
    scale = sample_anno['scale']

    # 임시 저장 (predict 함수 사용을 위해)
    temp_img_path = 'temp_sample.jpg'
    sample_image.save(temp_img_path)

    print(f"Predicting on Validation Sample Index: {idx}...")

    try:
        # 예측 수행 (Keypoints/Scale 전달하여 Crop 수행)
        # 반환되는 img_hg, img_sb는 Crop된 이미지입니다.
        img_hg, kp_hg = predict(model_hg, temp_img_path, keypoints=joints, scale=scale)
        img_sb, kp_sb = predict(model_sb, temp_img_path, keypoints=joints, scale=scale)

        # 시각화 레이아웃 설정
        fig = plt.figure(figsize=(12, 18))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

        # --- Top: Ground Truth Skeleton (Original Image) ---
        ax_gt = fig.add_subplot(gs[0, :])  # 첫 번째 행 전체 사용
        draw_skeleton_on_image(sample_image, gt_joints, ax=ax_gt)
        ax_gt.set_title("Ground Truth Skeleton (Original Image)")

        # --- Middle: Stacked Hourglass (Cropped Image) ---
        # Left: Keypoints
        ax_hg_kp = fig.add_subplot(gs[1, 0])
        draw_keypoints_on_image(img_hg, kp_hg, ax=ax_hg_kp)
        ax_hg_kp.set_title("StackedHourglass - Keypoints (Cropped)")

        # Left: Skeleton
        ax_hg_sk = fig.add_subplot(gs[2, 0])
        draw_skeleton_on_image(img_hg, kp_hg, ax=ax_hg_sk)
        ax_hg_sk.set_title("StackedHourglass - Skeleton (Cropped)")

        # --- Bottom: Simple Baseline (Cropped Image) ---
        # Right: Keypoints
        ax_sb_kp = fig.add_subplot(gs[1, 1])
        draw_keypoints_on_image(img_sb, kp_sb, ax=ax_sb_kp)
        ax_sb_kp.set_title("SimpleBaseline - Keypoints (Cropped)")

        # Right: Skeleton
        ax_sb_sk = fig.add_subplot(gs[2, 1])
        draw_skeleton_on_image(img_sb, kp_sb, ax=ax_sb_sk)
        ax_sb_sk.set_title("SimpleBaseline - Skeleton (Cropped)")

        plt.tight_layout()
        plt.show()

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
