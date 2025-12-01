import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 커스텀 모듈 임포트
from dataset import KittiDataset
from models import UNet, UNetPlusPlus
from losses import FocalDiceLoss
from visualize_augmentation import build_augmentation
from metrics import compute_iou, compute_boundary_iou
import compare_models


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_miou = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)

        # Deep Supervision 처리
        if isinstance(outputs, list):
            # UNet++ Deep Supervision: L1, L2, L3, L4 outputs
            weights = [0.1, 0.2, 0.3, 0.4]
            loss = 0
            for output, weight in zip(outputs, weights):
                l, _ = criterion(output, targets)
                loss += l * weight
            final_output = outputs[-1]
        else:
            loss, _ = criterion(outputs, targets)
            final_output = outputs

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics Calculation
        # Training에서는 속도를 위해 mIoU만 계산하고 Boundary IoU는 생략
        pred = torch.argmax(final_output, dim=1)
        miou, _ = compute_iou(pred, targets, num_classes=2, ignore_index=-100)

        running_loss += loss.item()
        running_miou += miou

        pbar.set_postfix({'loss': loss.item(), 'mIoU': miou})

    return running_loss / len(loader), running_miou / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_miou = 0.0
    running_boundary_iou = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            if isinstance(outputs, list):
                weights = [0.1, 0.2, 0.3, 0.4]
                loss = 0
                for output, weight in zip(outputs, weights):
                    l, _ = criterion(output, targets)
                    loss += l * weight
                final_output = outputs[-1]
            else:
                loss, _ = criterion(outputs, targets)
                final_output = outputs

            pred = torch.argmax(final_output, dim=1)

            # Validation에서는 Boundary IoU도 계산
            miou, _ = compute_iou(
                pred, targets, num_classes=2, ignore_index=-100)
            boundary_iou, _ = compute_boundary_iou(
                pred, targets, num_classes=2, ignore_index=-100)

            running_loss += loss.item()
            running_miou += miou
            running_boundary_iou += boundary_iou

            pbar.set_postfix({'val_loss': loss.item(), 'val_mIoU': miou})

    return running_loss / len(loader), running_miou / len(loader), running_boundary_iou / len(loader)


def run_training(model_name, model, train_loader, test_loader, num_epochs, device):
    print(f"\n{'='*20} Training {model_name} {'='*20}")

    os.makedirs("training", exist_ok=True)
    save_path = f"training/seg_model_{model_name.lower()}.pth"

    criterion = FocalDiceLoss(num_classes=2, ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    best_score = 0.0
    history = {
        'train_loss': [], 'train_miou': [],
        'val_loss': [], 'val_miou': [], 'val_boundary_iou': []
    }

    for epoch in range(num_epochs):
        train_loss, train_miou = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_miou, val_boundary_iou = validate(
            model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_miou)

        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_boundary_iou'].append(val_boundary_iou)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")
        print(
            f"Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Boundary IoU: {val_boundary_iou:.4f}")

        # Best Model Selection
        # mIoU를 우선으로 하되, Boundary IoU를 가중치로 더하여 종합 점수 계산
        # Score = mIoU + 0.5 * Boundary IoU
        current_score = val_miou + 0.5 * val_boundary_iou

        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), save_path)
            print(
                f"✅ Best model saved to {save_path} (Score: {best_score:.4f}, mIoU: {val_miou:.4f}, B-IoU: {val_boundary_iou:.4f})")

    return history


def plot_results(unet_hist, unetpp_hist):
    epochs = range(1, len(unet_hist['train_loss']) + 1)

    plt.figure(figsize=(18, 5))

    # Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, unet_hist['train_loss'], 'b--', label='UNet Train')
    plt.plot(epochs, unet_hist['val_loss'], 'b-', label='UNet Val')
    plt.plot(epochs, unetpp_hist['train_loss'], 'r--', label='UNet++ Train')
    plt.plot(epochs, unetpp_hist['val_loss'], 'r-', label='UNet++ Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # mIoU Plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, unet_hist['train_miou'], 'b--', label='UNet Train')
    plt.plot(epochs, unet_hist['val_miou'], 'b-', label='UNet Val')
    plt.plot(epochs, unetpp_hist['train_miou'], 'r--', label='UNet++ Train')
    plt.plot(epochs, unetpp_hist['val_miou'], 'r-', label='UNet++ Val')
    plt.title('mIoU Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()

    # Boundary IoU Plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, unet_hist['val_boundary_iou'], 'b-', label='UNet Val')
    plt.plot(epochs, unetpp_hist['val_boundary_iou'], 'r-', label='UNet++ Val')
    plt.title('Boundary IoU Comparison (Val)')
    plt.xlabel('Epochs')
    plt.ylabel('Boundary IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training/training_comparison.png')


def main():
    # 설정
    DATA_DIR = 'data_semantics/training'
    BATCH_SIZE = 2
    NUM_EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SPLIT_RATIO = 0.9  # 90% Train, 10% Val

    print(f"Device: {DEVICE}")

    # 데이터셋 준비 (Flexible Splitting)
    augmentation_train = build_augmentation(is_train=True)
    augmentation_test = build_augmentation(is_train=False)

    train_dataset = KittiDataset(
        DATA_DIR, is_train=True, augmentation=augmentation_train, num_classes=2, split_ratio=SPLIT_RATIO)
    test_dataset = KittiDataset(
        DATA_DIR, is_train=False, augmentation=augmentation_test, num_classes=2, split_ratio=SPLIT_RATIO)

    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(test_dataset)} images")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 1. UNet 학습
    unet_model = UNet(in_channels=3, num_classes=2).to(DEVICE)
    unet_history = run_training(
        "UNet", unet_model, train_loader, test_loader, NUM_EPOCHS, DEVICE)

    # GPU 메모리 해제 (첫 번째 모델 훈련 완료 후)
    print("\nReleasing UNet model from GPU memory...")
    unet_model.cpu()  # 모델을 CPU로 이동
    del unet_model    # 모델 객체 삭제
    torch.cuda.empty_cache()  # GPU 캐시 정리
    print("GPU memory cleared successfully\n")

    # 2. UNet++ 학습
    unetpp_model = UNetPlusPlus(
        in_channels=3, num_classes=2, deep_supervision=True).to(DEVICE)
    unetpp_history = run_training(
        "UNetPlusPlus", unetpp_model, train_loader, test_loader, NUM_EPOCHS, DEVICE)

    # 결과 비교 그래프
    plot_results(unet_history, unetpp_history)

    # 3. 자동 모델 비교 실행
    print("\n" + "="*40)
    print("Running Model Comparison...")
    print("="*40)
    try:
        compare_models.main()
    except Exception as e:
        print(f"Error running comparison: {e}")


if __name__ == '__main__':
    main()
