"""
Pose Estimation 결과 시각화 스크립트
두 모델(Hourglass, SimpleBaseline)의 예측 결과를 비교 시각화합니다.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os

# MPII 관절 인덱스
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


def find_max_coordinates(heatmaps):
    """히트맵에서 최대값 좌표를 추출"""
    H, W, C = heatmaps.shape
    flatten_heatmaps = heatmaps.reshape(-1, C)
    indices = torch.argmax(flatten_heatmaps, dim=0)
    y = indices // W
    x = indices % W
    return torch.stack([x, y], dim=1)


def extract_keypoints_from_heatmap(heatmaps):
    """히트맵에서 keypoint 좌표를 추출 (서브픽셀 정밀도)"""
    H, W, C = heatmaps.shape
    max_keypoints = find_max_coordinates(heatmaps)

    heatmaps_permuted = heatmaps.permute(2, 0, 1)
    padded = F.pad(heatmaps_permuted, (1, 1, 1, 1))
    padded_heatmaps = padded.permute(1, 2, 0)

    adjusted_keypoints = []
    for i, keypoint in enumerate(max_keypoints):
        max_x = int(keypoint[0].item()) + 1
        max_y = int(keypoint[1].item()) + 1

        patch = padded_heatmaps[max_y-1:max_y+2, max_x-1:max_x+2, i]
        patch = patch.clone()
        patch[1, 1] = 0
        flat_patch = patch.reshape(-1)
        index = torch.argmax(flat_patch).item()

        next_y = index // 3
        next_x = index % 3
        delta_y = (next_y - 1) / 4.0
        delta_x = (next_x - 1) / 4.0

        adjusted_x = keypoint[0].item() + delta_x
        adjusted_y = keypoint[1].item() + delta_y
        adjusted_keypoints.append((adjusted_x, adjusted_y))

    adjusted_keypoints = torch.tensor(adjusted_keypoints)
    adjusted_keypoints = torch.clamp(adjusted_keypoints, 0, H)
    normalized_keypoints = adjusted_keypoints / H
    return normalized_keypoints


def predict(model, image_path):
    """모델로 이미지에서 keypoint 예측"""
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    inputs = preprocess(image).unsqueeze(0)

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    if not isinstance(outputs, list):
        outputs = [outputs]

    heatmap_tensor = outputs[-1].squeeze(0).permute(1, 2, 0)
    heatmap = heatmap_tensor.detach().cpu()
    kp = extract_keypoints_from_heatmap(heatmap)

    return image, kp


def to_numpy_image(image):
    """이미지를 numpy 배열로 변환"""
    if torch.is_tensor(image):
        image_np = image.detach().cpu().numpy()
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
    elif not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
    return image_np


def draw_skeleton_comparison(image, kp_hg, kp_sb, title="", save_path=None):
    """두 모델의 스켈레톤을 나란히 비교"""
    image_np = to_numpy_image(image)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 이미지
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    # Hourglass 결과
    axes[1].imshow(image_np)
    joints_hg = []
    for joint in kp_hg:
        joint_x = joint[0] * image_np.shape[1]
        joint_y = joint[1] * image_np.shape[0]
        joints_hg.append((joint_x, joint_y))
        axes[1].scatter(joint_x, joint_y, s=40, c='red', marker='o', zorder=5, edgecolors='white', linewidths=1)
    
    for bone in MPII_BONES:
        j1, j2 = joints_hg[bone[0]], joints_hg[bone[1]]
        axes[1].plot([j1[0], j2[0]], [j1[1], j2[1]], linewidth=3, alpha=0.8, c='cyan')
    axes[1].set_title("Hourglass", fontsize=12, fontweight='bold', color='cyan')
    axes[1].axis('off')
    
    # SimpleBaseline 결과
    axes[2].imshow(image_np)
    joints_sb = []
    for joint in kp_sb:
        joint_x = joint[0] * image_np.shape[1]
        joint_y = joint[1] * image_np.shape[0]
        joints_sb.append((joint_x, joint_y))
        axes[2].scatter(joint_x, joint_y, s=40, c='red', marker='o', zorder=5, edgecolors='white', linewidths=1)
    
    for bone in MPII_BONES:
        j1, j2 = joints_sb[bone[0]], joints_sb[bone[1]]
        axes[2].plot([j1[0], j2[0]], [j1[1], j2[1]], linewidth=3, alpha=0.8, c='lime')
    axes[2].set_title("SimpleBaseline", fontsize=12, fontweight='bold', color='lime')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    plt.show()


def load_model(model_class, model_name, checkpoint_dir, device):
    """체크포인트에서 모델 로드"""
    model = model_class(num_heatmap=16)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}-best.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}-latest.pt')
    
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


if __name__ == "__main__":
    from model import StackedHourglassNetwork, SimpleBaseline
    
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(PROJECT_PATH, 'checkpoints')
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드
    print("\nLoading models...")
    model_hg = load_model(StackedHourglassNetwork, 'hourglass', CHECKPOINT_DIR, device)
    model_sb = load_model(SimpleBaseline, 'simplebaseline', CHECKPOINT_DIR, device)
    
    if model_hg is None or model_sb is None:
        print("Error: Could not load models!")
        exit(1)
    
    # 테스트 이미지 목록
    test_images = [
        os.path.join(PROJECT_PATH, 'samples.jpg'),
        os.path.join(PROJECT_PATH, 'samples1.jpg'),
    ]
    
    print("\nProcessing images...")
    
    # 각 이미지에 대해 예측 및 시각화
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        img_name = os.path.basename(img_path)
        print(f"\nProcessing: {img_name}")
        
        # Hourglass 예측
        image_hg, kp_hg = predict(model_hg, img_path)
        
        # SimpleBaseline 예측
        image_sb, kp_sb = predict(model_sb, img_path)
        
        # 비교 시각화
        title = f"Pose Estimation Comparison - {img_name}"
        save_path = os.path.join(PROJECT_PATH, f"result_{img_name.replace('.jpg', '.png')}")
        draw_skeleton_comparison(image_hg, kp_hg, kp_sb, title, save_path)
        
        print(f"  Hourglass keypoints: {kp_hg.shape}")
        print(f"  SimpleBaseline keypoints: {kp_sb.shape}")
    
    print("\n" + "="*50)
    print("Done! Results saved as PNG files.")
    print("="*50)