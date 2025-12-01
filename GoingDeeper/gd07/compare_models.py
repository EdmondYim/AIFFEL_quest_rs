import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize

# 커스텀 모듈
from dataset import KittiDataset
from models import UNet, UNetPlusPlus
from visualize_augmentation import build_augmentation
from metrics import compute_iou, compute_boundary_iou
from inference import decode_segmap, create_palette


NUM_CLASSES = 34
IGNORE_INDEX = -100


def evaluate_model(model, loader, device, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    model.eval()
    miou_list = []
    boundary_iou_list = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            if isinstance(outputs, list):
                outputs = outputs[-1]

            pred = torch.argmax(outputs, dim=1)

            miou, _ = compute_iou(pred, targets, num_classes, ignore_index)
            boundary_iou, _ = compute_boundary_iou(
                pred, targets, num_classes, ignore_index)

            miou_list.append(miou)
            boundary_iou_list.append(boundary_iou)

    return np.mean(miou_list), np.mean(boundary_iou_list)


def visualize_comparison(unet_model, unetpp_model, dataset, device, palette, num_samples=5, save_dir='comparison_results'):
    os.makedirs(save_dir, exist_ok=True)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]

        # Prepare input
        input_tensor = image_tensor.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            # UNet
            unet_out = unet_model(input_tensor)
            unet_pred = torch.argmax(unet_out, dim=1).squeeze().cpu().numpy()

            # UNet++
            unetpp_out = unetpp_model(input_tensor)
            if isinstance(unetpp_out, list):
                unetpp_out = unetpp_out[-1]
            unetpp_pred = torch.argmax(
                unetpp_out, dim=1).squeeze().cpu().numpy()

        # Prepare visualization
        # Image: (C, H, W) -> (H, W, C)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed (assuming [0,1] range)
        image_np = (image_np * 255).astype(np.uint8)

        mask_np = mask_tensor.cpu().numpy()

        # Decode segmentation maps
        gt_rgb = decode_segmap(mask_np, palette)
        unet_rgb = decode_segmap(unet_pred, palette)
        unetpp_rgb = decode_segmap(unetpp_pred, palette)

        # Plot
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(unet_rgb)
        plt.title("UNet Prediction")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(unetpp_rgb)
        plt.title("UNet++ Prediction")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{i}.png'))
        plt.close()

    print(f"Comparison images saved to {save_dir}")


def main():
    # 설정
    DATA_DIR = 'data_semantics/training'
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SPLIT_RATIO = 0.9
    palette = create_palette(NUM_CLASSES)

    print(f"Device: {DEVICE}")

    # 데이터셋 준비 (Test set - using validation split)
    test_aug = build_augmentation(is_train=False)
    test_dataset = KittiDataset(
        DATA_DIR, is_train=False, augmentation=test_aug, num_classes=NUM_CLASSES, split_ratio=SPLIT_RATIO)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    unet_path = 'training/seg_model_unet.pth'
    unetpp_path = 'training/seg_model_unetplusplus.pth'

    models_to_evaluate = {}

    # UNet 로드 시도
    if os.path.exists(unet_path):
        unet = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
        unet.load_state_dict(torch.load(unet_path, map_location=DEVICE))
        models_to_evaluate['UNet'] = unet
        print(f"Loaded UNet from {unet_path}")
    else:
        print(f"UNet checkpoint not found at {unet_path}")

    # UNet++ 로드 시도
    if os.path.exists(unetpp_path):
        unetpp = UNetPlusPlus(in_channels=3, num_classes=NUM_CLASSES,
                              deep_supervision=True).to(DEVICE)
        unetpp.load_state_dict(torch.load(unetpp_path, map_location=DEVICE))
        models_to_evaluate['UNet++'] = unetpp
        print(f"Loaded UNet++ from {unetpp_path}")
    else:
        print(f"UNet++ checkpoint not found at {unetpp_path}")

    if not models_to_evaluate:
        print("No models found to evaluate!")
        return

    # 정량적 평가
    results = {}
    print("\nStarting Evaluation...")

    result_str = f"{'='*50}\nModel Comparison Results\n{'='*50}\n"
    result_str += f"{'Model':<15} | {'mIoU':<10} | {'Boundary IoU':<15}\n"
    result_str += f"{'-'*50}\n"

    for name, model in models_to_evaluate.items():
        print(f"Evaluating {name}...")
        miou, biou = evaluate_model(model, test_loader, DEVICE)
        results[name] = {'mIoU': miou, 'Boundary IoU': biou}
        result_str += f"{name:<15} | {miou:.4f}     | {biou:.4f}\n"

    result_str += f"{'='*50}\n"
    print("\n" + result_str)

    with open('training/comparison_results.txt', 'w') as f:
        f.write(result_str)

    # 시각적 비교 (두 모델 다 있을 때만 side-by-side 비교가 의미 있음, 하나만 있으면 그것만 출력)
    print("\nGenerating visual comparisons...")

    os.makedirs('comparison_results', exist_ok=True)
    indices = np.random.choice(len(test_dataset), 10, replace=False)

    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = test_dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Image & GT
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        mask_np = mask_tensor.cpu().numpy()
        gt_rgb = decode_segmap(mask_np, palette)

        # Plot 설정
        num_cols = 2 + len(models_to_evaluate)
        plt.figure(figsize=(5 * num_cols, 5))

        plt.subplot(1, num_cols, 1)
        plt.imshow(image_np)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, num_cols, 2)
        plt.imshow(gt_rgb)
        plt.title("Ground Truth")
        plt.axis('off')

        col_idx = 3
        for name, model in models_to_evaluate.items():
            with torch.no_grad():
                out = model(input_tensor)
                if isinstance(out, list):
                    out = out[-1]
                pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
                pred_rgb = decode_segmap(pred, palette)

                plt.subplot(1, num_cols, col_idx)
                plt.imshow(pred_rgb)
                plt.title(f"{name} Prediction")
                plt.axis('off')
                col_idx += 1

        plt.tight_layout()
        plt.savefig(f'comparison_results/comparison_{i}.png')
        plt.close()

    print(f" Comparison images saved to comparison_results/")


if __name__ == '__main__':
    main()
