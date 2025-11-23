import torch
import torch.optim as optim
import argparse
import os
import time
import math
# tkinter 스레딩 문제를 피하기 위해 matplotlib 백엔드를 non-GUI로 설정
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from face_detector.model import SSD
from face_detector.dataset import WiderFaceDataset, default_box
from face_detector.utils import MultiBoxLoss, MultiStepWarmUpLR, encode_pt

def train_step(model, optimizer, criterion, inputs, labels, debug=False):
    model.train()
    optimizer.zero_grad()

    # 디버그: 입력 확인
    if debug:
        print(f"[DEBUG] Input - Has NaN: {torch.isnan(inputs).any()}, Has Inf: {torch.isinf(inputs).any()}")
        print(f"[DEBUG] Input - Min: {inputs.min().item():.4f}, Max: {inputs.max().item():.4f}")
        print(f"[DEBUG] Labels shape: {labels.shape}, Has NaN: {torch.isnan(labels).any()}")

    predictions = model(inputs)
    
    # 디버그: 모델 출력 확인
    if debug:
        print(f"[DEBUG] Predictions - Has NaN: {torch.isnan(predictions).any()}, Has Inf: {torch.isinf(predictions).any()}")
        print(f"[DEBUG] Predictions - Min: {predictions.min().item():.4f}, Max: {predictions.max().item():.4f}")
    
    loss_loc, loss_class, acc = criterion(labels, predictions)

    # 디버그: 손실 구성 요소 확인
    if debug:
        print(f"[DEBUG] Loss_loc: {loss_loc.item()}, Loss_class: {loss_class.item()}")
        print(f"[DEBUG] Loss_loc - Is NaN: {torch.isnan(loss_loc)}, Is Inf: {torch.isinf(loss_loc)}")
        print(f"[DEBUG] Loss_class - Is NaN: {torch.isnan(loss_class)}, Is Inf: {torch.isinf(loss_class)}")
        criterion.debug = False  # 디버그 실행 후 재설정

    total_loss = loss_loc + loss_class

    if torch.isnan(total_loss):
        return total_loss.item(), {'loc': loss_loc.item(), 'class': loss_class.item(), 'acc': acc.item()}, True

    total_loss.backward()

    # 그래디언트 폭발 방지를 위한 그래디언트 클리핑
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

    optimizer.step()

    return total_loss.item(), {'loc': loss_loc.item(), 'class': loss_class.item(), 'acc': acc.item()}, False

def evaluate(model, val_loader, criterion, device, boxes):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            
            # GPU에서 타겟 인코딩
            labels_encoded = []
            for target in targets:
                target = target.to(device)
                encoded = encode_pt(target, boxes)
                labels_encoded.append(encoded)
            labels = torch.stack(labels_encoded, 0)

            predictions = model(inputs)
            loss_loc, loss_class, acc = criterion(labels, predictions)
            
            total_loss += (loss_loc + loss_class).item()
            total_acc += acc.item()
            steps += 1

    return total_loss / steps if steps > 0 else 0.0, total_acc / steps if steps > 0 else 0.0

def collate_fn(batch):
    images = []
    targets = []
    for img, label in batch:
        images.append(img)
        targets.append(label)
    return torch.stack(images, 0), targets

def plot_metrics(train_losses, train_accs, val_losses, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 하이퍼파라미터
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    IMAGE_WIDTH = 320
    IMAGE_HEIGHT = 256
    IMAGE_LABELS = ['background', 'face']
    
    # 모델 초기화
    model = SSD(num_classes=len(IMAGE_LABELS), input_shape=(3, IMAGE_HEIGHT, IMAGE_WIDTH)).to(device)
    
    # GPU 최적화
    torch.backends.cudnn.benchmark = True
    
    # 데이터셋 초기화
    boxes = default_box(IMAGE_HEIGHT, IMAGE_WIDTH).to(device)
    
    # 학습 데이터셋
    train_dataset = WiderFaceDataset(args.data_path, split='train', boxes=None)
    train_loader = torch.utils.data.DataLoader(
        lr_rate=0.1,
        warmup_steps=5 * steps_per_epoch,
        min_lr=5e-5
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, nesterov=True)
    criterion = MultiBoxLoss(len(IMAGE_LABELS), neg_pos_ratio=3)

    # 지표 기록
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print("Starting training...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_acc = 0.0
        train_steps = 0
        
        for step, (inputs, targets) in enumerate(train_loader):
            # 학습률 업데이트
            current_step = epoch * steps_per_epoch + step
            lr = learning_rate(current_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            inputs = inputs.to(device)
            
            # GPU에서 타겟 인코딩
            labels_encoded = []
            for target in targets:
                target = target.to(device)
                encoded = encode_pt(target, boxes)
                labels_encoded.append(encoded)
            labels = torch.stack(labels_encoded, 0)

            load_t0 = time.time()
            total_loss, losses, is_nan = train_step(model, optimizer, criterion, inputs, labels, debug=False)
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            
            if is_nan:
                print(f"\n\n[ERROR] NaN loss detected at Epoch {epoch+1} Batch {step+1}!")
                print(f"Loc: {losses['loc']}, Class: {losses['class']}")
                print(f"Re-running with debug mode enabled...\n")
                
                # 디버그 모드로 재실행
                criterion.debug = True
                model.eval()
                with torch.no_grad():
                    _, _, _ = train_step(model, optimizer, criterion, inputs, labels, debug=True)
                
                print(f"\n[ERROR] Training stopped due to NaN. Please check the debug output above.")
                raise RuntimeError("NaN loss encountered") 
            
            epoch_loss += total_loss
            epoch_acc += losses['acc']
            train_steps += 1

            if step % 10 == 0:
                print(f"\rEpoch: {epoch + 1}/{EPOCHS} | Batch {step + 1}/{steps_per_epoch} | Time {batch_time:.3f}s | Loss: {total_loss:.6f} | Acc: {losses['acc']:.4f} | Loc: {losses['loc']:.6f} | Class: {losses['class']:.6f}", end='', flush=True)
        
        print() # Newline after epoch
        
        # 평균 학습 지표 계산
        avg_train_loss = epoch_loss / train_steps if train_steps > 0 else 0.0
        avg_train_acc = epoch_acc / train_steps if train_steps > 0 else 0.0
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        # 검증
        print("Validating...")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, boxes)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 그래프 그리기
        plot_metrics(train_losses, train_accs, val_losses, val_accs, os.path.join(args.checkpoint_dir, 'training_curves.png'))
        
        # 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'ssd_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSD 얼굴 감지기 학습')
    parser.add_argument('--data_path', type=str, default=r'C:\Users\C\Desktop\detec\widerface', help='widerface 데이터셋 경로')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=80, help='에포크 수')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩을 위한 워커 수')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='체크포인트를 저장할 디렉토리')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    main(args)