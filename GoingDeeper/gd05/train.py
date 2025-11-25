"""
RetinaNet 훈련 스크립트
KITTI 데이터셋을 사용한 객체 탐지 모델 훈련
"""

import os
import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용

from dataset import KITTIDataset
from dataloader import get_train_dataloader, get_val_dataloader
from model import RetinaNet, RetinaNetLoss, AnchorBox, get_resnet50_backbone


# ============================================================================
# Label Encoder for Training
# ============================================================================

class LabelEncoder:
    """
    Ground truth boxes와 labels를 anchor boxes에 매칭하여 학습용 타겟 생성
    """
    
    def __init__(self, box_variance=[0.1, 0.1, 0.2, 0.2]):
        self._anchor_box = AnchorBox()
        self._box_variance = torch.tensor(box_variance, dtype=torch.float32)
    
    def _compute_iou(self, boxes1, boxes2):
        """IoU 계산 (boxes format: [x_min, y_min, x_max, y_max])"""
        # Intersection
        lu = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rd = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        intersection = torch.clamp(rd - lu, min=0.0)
        intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
        
        # Areas
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Union
        union_area = boxes1_area[:, None] + boxes2_area - intersection_area
        
        return intersection_area / torch.clamp(union_area, min=1e-8)
    
    def _match_anchor_boxes(self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        """Anchor boxes를 ground truth boxes에 매칭"""
        # Convert anchors from [cx, cy, w, h] to [x_min, y_min, x_max, y_max]
        anchor_corners = torch.cat([
            anchor_boxes[:, :2] - anchor_boxes[:, 2:] / 2.0,
            anchor_boxes[:, :2] + anchor_boxes[:, 2:] / 2.0
        ], dim=-1)
        
        iou_matrix = self._compute_iou(anchor_corners, gt_boxes)
        max_iou, matched_gt_idx = torch.max(iou_matrix, dim=1)
        
        positive_mask = max_iou >= match_iou
        negative_mask = max_iou < ignore_iou
        ignore_mask = ~(positive_mask | negative_mask)
        
        return matched_gt_idx, positive_mask.float(), ignore_mask.float()
    
    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Anchor box와 GT box 간의 offset 계산"""
        # Convert GT boxes to [cx, cy, w, h]
        matched_gt_cxcywh = torch.cat([
            (matched_gt_boxes[:, :2] + matched_gt_boxes[:, 2:]) / 2.0,
            matched_gt_boxes[:, 2:] - matched_gt_boxes[:, :2]
        ], dim=-1)
        
        box_target = torch.cat([
            (matched_gt_cxcywh[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
            torch.log(matched_gt_cxcywh[:, 2:] / anchor_boxes[:, 2:])
        ], dim=-1)
        
        box_target = box_target / self._box_variance.to(box_target.device)
        
        return box_target
    
    def encode_batch(self, images, batch_boxes, batch_labels):
        """
        배치 데이터를 인코딩
        
        Args:
            images: (B, C, H, W) 이미지
            batch_boxes: List of (N, 4) boxes per image
            batch_labels: List of (N,) labels per image
        
        Returns:
            encoded_labels: (B, num_anchors, 5) encoded targets
        """
        batch_size = images.size(0)
        image_height, image_width = images.size(2), images.size(3)
        
        anchor_boxes = self._anchor_box.get_anchors(image_height, image_width)
        anchor_boxes = anchor_boxes.to(images.device)
        
        encoded_labels = []
        
        for i in range(batch_size):
            gt_boxes = batch_boxes[i].to(images.device)
            cls_ids = batch_labels[i].to(images.device).float()
            
            if len(gt_boxes) == 0:
                # No objects in this image
                label = torch.zeros((len(anchor_boxes), 5), device=images.device)
                label[:, 4] = -1.0  # All negative
                encoded_labels.append(label)
                continue
            
            matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
                anchor_boxes, gt_boxes
            )
            
            matched_gt_boxes = gt_boxes[matched_gt_idx]
            box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
            
            matched_gt_cls_ids = cls_ids[matched_gt_idx]
            cls_target = torch.where(positive_mask != 1.0, -1.0, matched_gt_cls_ids)
            cls_target = torch.where(ignore_mask == 1.0, -2.0, cls_target)
            
            label = torch.cat([box_target, cls_target.unsqueeze(-1)], dim=-1)
            encoded_labels.append(label)
        
        return torch.stack(encoded_labels, dim=0)


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, loss_fn, label_encoder, device, epoch):
    """한 에폭 훈련"""
    model.train()
    
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        boxes = [t['boxes'] for t in targets]
        labels = [t['labels'] for t in targets]
        
        # Encode labels
        encoded_labels = label_encoder.encode_batch(images, boxes, labels)
        
        # Forward
        predictions = model(images)
        
        # Loss (이미 스칼라 반환)
        loss = loss_fn(encoded_labels, predictions)
        
        # Accuracy (for positive anchors only)
        with torch.no_grad():
            positive_mask = encoded_labels[:, :, 4] >= 0
            if positive_mask.sum() > 0:
                cls_predictions = predictions[:, :, 4:]
                cls_labels = encoded_labels[:, :, 4].long()
                
                pred_classes = torch.argmax(cls_predictions[positive_mask], dim=-1)
                true_classes = cls_labels[positive_mask]
                
                running_correct += (pred_classes == true_classes).sum().item()
                running_total += positive_mask.sum().item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Update progress bar
        current_acc = 100.0 * running_correct / running_total if running_total > 0 else 0.0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * running_correct / running_total if running_total > 0 else 0.0
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, loss_fn, label_encoder, device, epoch):
    """검증"""
    model.eval()
    
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            boxes = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]
            
            # Encode labels
            encoded_labels = label_encoder.encode_batch(images, boxes, labels)
            
            # Forward
            predictions = model(images)
            
            # Loss (이미 스칼라 반환)
            loss = loss_fn(encoded_labels, predictions)
            
            # Accuracy (for positive anchors only)
            positive_mask = encoded_labels[:, :, 4] >= 0
            if positive_mask.sum() > 0:
                cls_predictions = predictions[:, :, 4:]
                cls_labels = encoded_labels[:, :, 4].long()
                
                pred_classes = torch.argmax(cls_predictions[positive_mask], dim=-1)
                true_classes = cls_labels[positive_mask]
                
                running_correct += (pred_classes == true_classes).sum().item()
                running_total += positive_mask.sum().item()
            
            running_loss += loss.item()
            
            current_acc = 100.0 * running_correct / running_total if running_total > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * running_correct / running_total if running_total > 0 else 0.0
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """체크포인트 로드"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    
    return epoch, loss


def plot_training_history(history, save_dir):
    """훈련 히스토리 시각화"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved: {plot_path}")


def save_history_to_csv(history, save_dir):
    """훈련 히스토리를 CSV로 저장"""
    csv_path = os.path.join(save_dir, 'training_history.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_loss'][i],
                history['val_acc'][i]
            ])
    
    print(f"Training history CSV saved: {csv_path}")


# ============================================================================
# Main Training
# ============================================================================

def main(args):
    """메인 훈련 함수"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Dataset & DataLoader
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    
    train_loader = get_train_dataloader(
        img_dir=args.train_img_dir,
        label_dir=args.train_label_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        shuffle=True
    )
    
    val_loader = get_val_dataloader(
        img_dir=args.val_img_dir,
        label_dir=args.val_label_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    print("\n" + "=" * 60)
    print("Initializing model...")
    print("=" * 60)
    
    backbone = get_resnet50_backbone(pretrained=args.pretrained)
    model = RetinaNet(num_classes=args.num_classes, backbone=backbone)
    model = model.to(device)
    
    print(f"Model created with {args.num_classes} classes")
    
    # Loss & Optimizer
    loss_fn = RetinaNetLoss(num_classes=args.num_classes)
    label_encoder = LabelEncoder()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Load checkpoint if exists
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'last.pth')
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch += 1
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.img_size}")
    print("=" * 60 + "\n")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, label_encoder, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, loss_fn, label_encoder, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Time: {epoch_time:.1f}s")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, 'last.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_checkpoint_path)
            print(f"★ New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        print("-" * 60)
    
    # Save training history
    print("\n" + "=" * 60)
    print("Saving training history...")
    save_history_to_csv(history, args.checkpoint_dir)
    plot_training_history(history, args.checkpoint_dir)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print("=" * 60)
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaNet Training on KITTI')
    
    # Data paths
    parser.add_argument('--train_img_dir', type=str, 
                       default='archive/data_object_image_2/training/image_2',
                       help='Training images directory')
    parser.add_argument('--train_label_dir', type=str,
                       default='archive/data_object_label_2/training/label_2',
                       help='Training labels directory')
    parser.add_argument('--val_img_dir', type=str,
                       default='archive/data_object_image_2/training/image_2',
                       help='Validation images directory')
    parser.add_argument('--val_label_dir', type=str,
                       default='archive/data_object_label_2/training/label_2',
                       help='Validation labels directory')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=8,
                       help='Number of object classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ResNet50 backbone')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, nargs=2, default=[384, 1280],
                       help='Image size (height width)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # Checkpoint & logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='TensorBoard log directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    main(args)
