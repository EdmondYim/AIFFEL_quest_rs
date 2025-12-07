"""
Hourglass vs SimpleBaseline 모델 학습 및 비교 스크립트

사용법:
    python run_training.py --model hourglass --epochs 5
    python run_training.py --model simplebaseline --epochs 5
    python run_training.py --model both --epochs 5
"""

import os
import sys
import argparse
import torch
import time

# 현재 디렉토리의 모듈 직접 import
from model import StackedHourglassNetwork, SimpleBaseline
from train import train
import train as train_module

# 설정
TRAIN_JSON = 'archive/train.json'
VAL_JSON = 'archive/validation.json'
IMAGE_DIR = 'archive/mpii_human_pose_v1/images'
MODEL_SAVE_PATH = 'checkpoints'

# 학습 파라미터
NUM_HEATMAP = 16
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def check_gpu():
    """GPU 상태 확인"""
    print("\n" + "="*60)
    print("GPU Configuration")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA Available: False (Using CPU)")
    print("="*60 + "\n")


def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model_type, epochs, batch_size=BATCH_SIZE):
    """
    단일 모델 학습
    
    Args:
        model_type: 'hourglass' or 'simplebaseline'
        epochs: 학습 epoch 수
        batch_size: 배치 크기
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*60}")
    
    # 모델 생성
    if model_type == 'hourglass':
        model = StackedHourglassNetwork(num_heatmap=NUM_HEATMAP)
        model_name = 'hourglass'
    elif model_type == 'simplebaseline':
        model = SimpleBaseline(num_heatmap=NUM_HEATMAP)
        model_name = 'simplebaseline'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 파라미터 수 출력
    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # MODEL_PATH 설정
    train_module.MODEL_PATH = MODEL_SAVE_PATH
    
    # 체크포인트 디렉토리 생성
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 학습 시작 시간
    start_time = time.time()
    
    # 학습 실행
    history = train(
        model=model,
        model_name=model_name,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
        num_heatmap=NUM_HEATMAP,
        batch_size=batch_size,
        train_annotation_file=TRAIN_JSON,
        val_annotation_file=VAL_JSON,
        image_dir=IMAGE_DIR
    )
    
    # 학습 종료 시간
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} Training Complete!")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Time per Epoch: {total_time/epochs/60:.2f} minutes")
    print(f"{'='*60}\n")
    
    return history, total_time


def compare_models(history_hg, history_sb, time_hg, time_sb):
    """두 모델 결과 비교"""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # 최종 결과
    print("\n[Final Results]")
    print(f"{'Metric':<20} {'Hourglass':>15} {'SimpleBaseline':>15}")
    print("-" * 52)
    
    if history_hg and len(history_hg['val_loss']) > 0:
        hg_val_loss = history_hg['val_loss'][-1]
        hg_val_acc = history_hg['val_acc'][-1]
    else:
        hg_val_loss = float('nan')
        hg_val_acc = float('nan')
    
    if history_sb and len(history_sb['val_loss']) > 0:
        sb_val_loss = history_sb['val_loss'][-1]
        sb_val_acc = history_sb['val_acc'][-1]
    else:
        sb_val_loss = float('nan')
        sb_val_acc = float('nan')
    
    print(f"{'Val Loss':<20} {hg_val_loss:>15.4f} {sb_val_loss:>15.4f}")
    print(f"{'Val Accuracy':<20} {hg_val_acc:>15.4f} {sb_val_acc:>15.4f}")
    print(f"{'Training Time (min)':<20} {time_hg/60:>15.2f} {time_sb/60:>15.2f}")
    
    # 결론
    print("\n[Conclusion]")
    if hg_val_acc > sb_val_acc:
        print("  → Hourglass achieved higher validation accuracy")
    elif sb_val_acc > hg_val_acc:
        print("  → SimpleBaseline achieved higher validation accuracy")
    else:
        print("  → Both models achieved similar accuracy")
    
    if time_hg < time_sb:
        print(f"  → Hourglass was {time_sb/time_hg:.2f}x faster")
    else:
        print(f"  → SimpleBaseline was {time_hg/time_sb:.2f}x faster")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Pose Estimation Models')
    parser.add_argument('--model', type=str, default='both',
                        choices=['hourglass', 'simplebaseline', 'both'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # GPU 확인
    check_gpu()
    
    history_hg = None
    history_sb = None
    time_hg = 0
    time_sb = 0
    
    if args.model in ['hourglass', 'both']:
        history_hg, time_hg = train_model('hourglass', args.epochs, args.batch_size)
    
    if args.model in ['simplebaseline', 'both']:
        history_sb, time_sb = train_model('simplebaseline', args.epochs, args.batch_size)
    
    if args.model == 'both':
        compare_models(history_hg, history_sb, time_hg, time_sb)
    
    print("Training complete! Checkpoints saved to:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
