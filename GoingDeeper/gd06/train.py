import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TARGET_CHARACTERS, LabelConverter, MJDataset, collate_fn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Data paths
data_dir = 'data_lmdb_release/data_lmdb_release/training/MJ'
TRAIN_DATA_PATH = data_dir + '/MJ_train'
VALID_DATA_PATH = data_dir + '/MJ_valid'
TEST_DATA_PATH = data_dir + '/MJ_test'

# Hyperparameters
MAX_TEXT_LEN = 22
IMG_SIZE = (100, 32)
BATCH_SIZE = 128


class CRNN(nn.Module):
    def __init__(self, num_chars, img_height=32, img_width=100):
        super(CRNN, self).__init__()
        self.num_chars = num_chars

        # (3, H, W) -> (64, H, W)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # (64, H/2, W/2)

        # (64, H/2, W/2) -> (128, H/2, W/2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # (128, H/4, W/4)

        # (128, H/4, W/4) -> (256, H/4, W/4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d((1, 2))  # (256, H/4, W/8)

        # (256, H/4, W/8) -> (512, H/4, W/8)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((1, 2))  # (512, H/4, W/16)

        # (512, H/4, W/16) -> (512, (H/4)-1, (W/16)-1) conv(2,2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2))

        # Bi-LSTM
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        # 최종 fc
        self.fc = nn.Linear(512, self.num_chars)

    def forward(self, x):
        # (B,3,32,100)
        x = F.relu(self.conv1(x))      # -> (B,64,32,100)
        x = self.pool1(x)             # -> (B,64,16,50)
        x = F.relu(self.conv2(x))      # -> (B,128,16,50)
        x = self.pool2(x)             # -> (B,128,8,25)
        x = F.relu(self.conv3(x))      # -> (B,256,8,25)
        x = F.relu(self.conv4(x))      # -> (B,256,8,25)
        x = self.pool3(x)             # -> (B,256,8,12) (25->12)
        x = F.relu(self.conv5(x))      # -> (B,512,8,12)
        x = self.bn5(x)
        x = F.relu(self.conv6(x))      # -> (B,512,8,12)
        x = self.bn6(x)
        x = self.pool4(x)             # -> (B,512,8,6)
        x = F.relu(self.conv7(x))      # -> (B,512,7,5) (8->7, 6->5)

        b, c, h, w = x.size()
        # 시퀀스 길이 = h*w
        x = x.view(b, c, h * w)  # (B,512,35)
        x = x.permute(0, 2, 1)   # (B,35,512)

        # LSTM
        x, _ = self.lstm1(x)    # (B,35,512)
        x, _ = self.lstm2(x)    # (B,35,512)

        # 최종 FC
        x = self.fc(x)          # (B,35,num_chars)

        # PyTorch의 CTCLoss를 위해선 (T,B,C) 형태가 일반적
        # 여기서는 (B,T,C) -> (T,B,C)
        x = x.permute(1, 0, 2)  # (35,B,num_chars)
        
        # CTCLoss는 log probabilities를 기대함
        x = F.log_softmax(x, dim=2)
        return x


def run_training(
        train_loader,
        valid_loader,
        model,
        optimizer,
        criterion,
        patience=2,
        epochs=20,
        checkpoint_path="ocr/model_checkpoint.pth"
        ):

    best_val_loss = float('inf')
    patience_counter = 0

    print("학습시작!...")

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for idx, (imgs, labels_padded, input_lengths, label_lengths, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels_padded = labels_padded.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)  # (T,B,C)

            # CTCLoss는 입력 (T,B,C), 타겟 (B,S), input_length, target_length
            loss = criterion(outputs, labels_padded, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for idx, (imgs, labels_padded, input_lengths, label_lengths, _) in enumerate(valid_loader):
                imgs = imgs.to(device)
                labels_padded = labels_padded.to(device)
                input_lengths = input_lengths.to(device)
                label_lengths = label_lengths.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels_padded, input_lengths, label_lengths)
                valid_loss += loss.item()
                valid_batches += 1


        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss/train_batches:.4f}, val_loss={valid_loss/valid_batches:.4f}")

        # 체크포인트 저장
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model improved. Saved at {checkpoint_path}")
        else:
            patience_counter += 1

        # EarlyStopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model


if __name__ == "__main__":
    # Label converter 초기화
    label_converter = LabelConverter(TARGET_CHARACTERS)
    num_chars = len(label_converter.character) + 1  # +1 for blank (CTC blank=0)
    print(f"Number of classes (including blank): {num_chars}")

    # 데이터셋 및 데이터로더 생성
    train_dataset = MJDataset(TRAIN_DATA_PATH, label_converter=label_converter, img_size=IMG_SIZE, character=TARGET_CHARACTERS)
    valid_dataset = MJDataset(VALID_DATA_PATH, label_converter=label_converter, img_size=IMG_SIZE, character=TARGET_CHARACTERS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=False)

    # 모델 초기화
    model = CRNN(num_chars=num_chars).to(device)

    # Loss 및 optimizer
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)

    # 학습 실행
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = run_training(
        train_loader,
        valid_loader,
        model,
        optimizer,
        criterion,
        patience=2,
        epochs=20,
        checkpoint_path=f"{checkpoint_dir}/model_checkpoint.pth"
    )
