import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.cuda.amp import autocast, GradScaler
from dataset import create_dataloader

MODEL_PATH = None

class Trainer(object):
    def __init__(self,
                 model,
                 epochs,
                 global_batch_size,
                 initial_learning_rate,
                 model_name='model'):
        """
        - model: 학습시킬 PyTorch 모델(nn.Module)
        - epochs: 전체 학습 epoch 수
        - global_batch_size: 전체 배치 크기 (loss 계산 시 사용)
        - initial_learning_rate: 초기 학습률
        - model_name: 모델 저장을 위한 식별자
        """
        self.model = model
        self.epochs = epochs
        self.global_batch_size = global_batch_size
        self.model_name = model_name

        self.loss_object = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_learning_rate)

        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.patience_count = 0
        self.max_patience = 3

        self.best_model = None

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.scaler = GradScaler()

        if torch.cuda.device_count() > 1:
            print(f"멀티 GPU 사용 (GPU 개수: {torch.cuda.device_count()})")
            self.model = nn.DataParallel(self.model)
        else:
            print("단일 GPU 혹은 CPU 사용")

    def lr_decay(self):
        if self.patience_count >= self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_learning_rate

    def compute_loss(self, labels, outputs):
        loss = 0
        for output in outputs:
            # 히트맵에서 값이 있는 부분(peak)에 더 높은 가중치를 부여하여 학습을 돕습니다.
            # labels > 0 인 부분(즉, 가우시안 분포가 있는 영역)에 가중치 82 (1 + 81)를 적용하고,
            # 배경(0)에는 가중치 1을 적용합니다.
            weights = (labels > 0).float() * 81 + 1
            squared_error = (labels - output) ** 2
            weighted_error = squared_error * weights
            loss += weighted_error.mean() / self.global_batch_size
        return loss

    def get_max_preds(self, batch_heatmaps):
        batch_size, num_channels, h, w = batch_heatmaps.shape
        heatmaps_reshaped = batch_heatmaps.view(batch_size, num_channels, -1)
        idx = torch.argmax(heatmaps_reshaped, 2)
        maxvals = torch.amax(heatmaps_reshaped, 2).view(batch_size, num_channels, 1)
        idx = idx.view(batch_size, num_channels, 1)
        preds = torch.cat((idx % w, idx // w), dim=2).float()
        return preds, maxvals

    def compute_accuracy(self, output, target, threshold=3.0):
        with torch.no_grad():
            batch_size, num_channels, h, w = output.shape
            preds, _ = self.get_max_preds(output)
            gt, _ = self.get_max_preds(target)
            dist = torch.norm(preds - gt, dim=2)
            acc_matrix = (dist <= threshold).float()
            avg_acc = acc_matrix.mean().item()
        return avg_acc

    def train_step(self, images, labels, device):
        self.model.train()
        images = images.to(device)
        labels = labels.to(device)
        self.optimizer.zero_grad()

        with autocast():
            outputs = self.model(images)
            loss = self.compute_loss(labels, outputs)

        acc = self.compute_accuracy(outputs[-1], labels)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item(), acc

    def val_step(self, images, labels, device):
        self.model.eval()
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = self.model(images)
            loss = self.compute_loss(labels, outputs)
            acc = self.compute_accuracy(outputs[-1], labels)
        return loss.item(), acc

    def save_checkpoint(self, filename):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            print(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            
            # Load state dict
            # DataParallel로 저장된 경우와 아닌 경우 처리
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
                
            # Load history
            self.history = checkpoint['history']
            
            # Determine start epoch
            start_epoch = len(self.history['train_loss'])
            print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
            return start_epoch
        else:
            print(f"No checkpoint found at '{filename}'")
            return 0

    def run(self, train_loader, val_loader, device, start_epoch=0):
        print(f"Total batches in train_loader: {len(train_loader)}")
        print(f"Total batches in val_loader: {len(val_loader)}")

        # Resume logic: if start_epoch >= self.epochs, training is already done.
        if start_epoch >= self.epochs:
            print(f"Training already completed (Current epoch {start_epoch} >= Target epochs {self.epochs}). Skipping training.")
            return self.history

        for epoch in range(start_epoch + 1, self.epochs + 1):
            epoch_start_time = time.time()
            self.lr_decay()
            print(f"Start epoch {epoch} with learning rate {self.current_learning_rate:.6f}")

            # Training
            total_train_loss = 0.0
            total_train_acc = 0.0
            num_train_batches = 0
            for images, labels in train_loader:
                batch_start_time = time.time()
                batch_loss, batch_acc = self.train_step(images, labels, device)
                batch_duration = time.time() - batch_start_time

                total_train_loss += batch_loss
                total_train_acc += batch_acc
                num_train_batches += 1

                # 로그 출력: 처음 5개 혹은 100배치마다 출력
                if num_train_batches <= 5 or num_train_batches % 100 == 0:
                    print(f"[Train] batch {num_train_batches} loss {batch_loss:.4f} acc {batch_acc:.4f} time {batch_duration:.4f}s")

                # # 테스트 모드: 5배치만 돌고 종료
                # if num_train_batches >= 5:
                #     print("Test mode: Stopping training early for this epoch.")
                #     break

            train_loss = total_train_loss / num_train_batches
            train_acc = total_train_acc / num_train_batches
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            print(f"Epoch {epoch} train loss {train_loss:.4f} acc {train_acc:.4f}")

            # Validation
            total_val_loss = 0.0
            total_val_acc = 0.0
            num_val_batches = 0
            for images, labels in val_loader:
                batch_start_time = time.time()
                batch_loss, batch_acc = self.val_step(images, labels, device)
                batch_duration = time.time() - batch_start_time

                num_val_batches += 1

                # 로그 출력: 처음 5개 혹은 100배치마다 출력
                if num_val_batches <= 5 or num_val_batches % 100 == 0:
                    print(f"[Val] batch {num_val_batches} loss {batch_loss:.4f} acc {batch_acc:.4f} time {batch_duration:.4f}s")

                if not math.isnan(batch_loss):
                    total_val_loss += batch_loss
                    total_val_acc += batch_acc
                else:
                    num_val_batches -= 1

                # # 테스트 모드: 5배치만 돌고 종료
                # if num_val_batches >= 5:
                #     print("Test mode: Stopping validation early.")
                #     break

            if num_val_batches > 0:
                val_loss = total_val_loss / num_val_batches
                val_acc = total_val_acc / num_val_batches
            else:
                val_loss = float('nan')
                val_acc = 0.0

            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            print(f"Epoch {epoch} val loss {val_loss:.4f} acc {val_acc:.4f}")

            # Epoch duration
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch} finished in {epoch_duration:.4f}s | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

            # 1. 최신 모델 저장
            latest_model_name = os.path.join(MODEL_PATH, f'{self.model_name}-latest.pt')
            self.save_checkpoint(latest_model_name)

            # 2. 최고 성능 모델 저장
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                best_model_name = os.path.join(MODEL_PATH, f'{self.model_name}-best.pt')
                self.save_checkpoint(best_model_name)
                self.best_model = best_model_name

            self.last_val_loss = val_loss

        return self.history

def train(model, model_name, epochs, learning_rate, num_heatmap, batch_size, train_annotation_file, val_annotation_file, image_dir):
    """
    - model: 학습시킬 모델 인스턴스
    - model_name: 모델 이름 (저장 시 파일명에 사용)
    - epochs: 전체 학습 epoch 수
    - learning_rate: 초기 학습률
    - num_heatmap: 생성할 heatmap 개수
    - batch_size: 배치 크기
    - train_annotation_file: train.json 파일 경로
    - val_annotation_file: validation.json 파일 경로
    - image_dir: 이미지 파일들이 저장된 디렉토리 경로
    """
    global_batch_size = batch_size

    train_loader = create_dataloader(train_annotation_file, image_dir, batch_size, num_heatmap, is_train=True)
    val_loader = create_dataloader(val_annotation_file, image_dir, batch_size, num_heatmap, is_train=False)

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Trainer 초기화 (model_name 전달)
    trainer = Trainer(
        model,
        epochs,
        global_batch_size,
        initial_learning_rate=learning_rate,
        model_name=model_name
    )

    # Resume Logic: Check for existing checkpoint
    latest_checkpoint_path = os.path.join(MODEL_PATH, f'{model_name}-latest.pt')
    start_epoch = 0
    if os.path.exists(latest_checkpoint_path):
        print(f"Found existing checkpoint for {model_name}. Attempting to resume...")
        start_epoch = trainer.load_checkpoint(latest_checkpoint_path)
    else:
        print(f"No existing checkpoint found for {model_name}. Starting from scratch.")

    print(f"Start training {model_name} from epoch {start_epoch + 1}...")
    return trainer.run(train_loader, val_loader, device, start_epoch=start_epoch)
