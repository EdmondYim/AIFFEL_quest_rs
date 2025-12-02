# 보고서

### 1. KITTI 데이터셋 구성

```python
def build_augmentation(is_train=True):
    if is_train:    # 훈련용 데이터일 경우
        return Compose([
            HorizontalFlip(p=0.5),    # 50%의 확률로 좌우대칭
            ColorJitter(              # 채도, 명도, 대비 변경 (image만 적용)
                brightness=0.2,       # 명도 ±20%
                contrast=0.2,         # 대비 ±20%
                saturation=0.2,       # 채도 ±20%
                hue=0.1,              # 색조 ±10%
                p=0.8                 # 80% 확률로 적용
            ),
            Resize(                   # 원본 비율 유지 (약 3:1), U-Net에 최적화된 크기
                width=768,
                height=256
            )
        ], additional_targets={'mask': 'mask'})  # mask도 함께 transform
    return Compose([      # 테스트용 데이터일 경우에는 768x256으로 resize만 수행합니다.
        Resize(
            width=768,
            height=256
        )
    ], additional_targets={'mask': 'mask'})  # mask도 함께 transform
```

- **명도(brightness)**: ±20% 변경
- **대비(contrast)**: ±20% 변경
- **채도(saturation)**: ±20% 변경
- **색조(hue)**: ±10% 변경
- **적용 확률**: 80%
- 데이터셋의 3.3:1 비율 유지하면서 (256, 768)로 resize

### 2. U-Net 모델 훈련

![1회차.png](pics/1%ED%9A%8C%EC%B0%A8.png)

![2회차.png](pics/2%ED%9A%8C%EC%B0%A8.png)

![3회차.png](pics/3%ED%9A%8C%EC%B0%A8.png)

UNet (3 runs)

- mIoU 평균: 0.9315
- Boundary IoU 평균: 0.7185

UNet++ (3 runs)

- mIoU 평균: 0.9292
- Boundary IoU 평균: 0.7258

### 3. 세그멘테이션 결과 이미지

![comparison_0.png](pics/comparison_0.png)

![comparison_1.png](pics/comparison_1.png)

![comparison_2.png](pics/comparison_2.png)

![comparison_3.png](pics/comparison_3.png)

![comparison_4.png](pics/comparison_4.png)

U-Net을 통한 세그멘테이션 작업이 정상적으로 진행 되었음을 확인하였다.

KITTI 데이터셋 구성, U-Net 모델 훈련, 결과물 시각화의 한 사이클이 정상 수행되어 세그멘테이션 결과 이미지를 제출하였다.

### 4. U-Net++ 구현 코드

```python
class UNetPlusPlus(nn.Module):
    """
    참고 사이트
    https://arxiv.org/abs/1807.10165
    https://github.com/ZJUGiveLab/UNet-Version

    Nested skip pathways를 통해 gradient flow 개선 및 성능 향상
    여러 단계의 feature fusion을 통해 더 정확한 segmentation 수행
    """
    def __init__(self, in_channels=3, num_classes=2, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        filters = [64, 128, 256, 512, 1024]

        # 인코더 (column 0)
        self.conv0_0 = DoubleConv(in_channels, filters[0])
        self.pool0 = nn.MaxPool2d(2)

        self.conv1_0 = DoubleConv(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_0 = DoubleConv(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_0 = DoubleConv(filters[2], filters[3])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_0 = DoubleConv(filters[3], filters[4])

        # Nested skip pathways (중첩된 skip 경로)
        # Column 1
        self.up1_0 = NestedUp(filters[1], filters[0], n_concat=2)
        self.up2_1 = NestedUp(filters[2], filters[1], n_concat=2)
        self.up3_2 = NestedUp(filters[3], filters[2], n_concat=2)
        self.up4_3 = NestedUp(filters[4], filters[3], n_concat=2)

        # Column 2
        self.up1_1 = NestedUp(filters[1], filters[0], n_concat=3)
        self.up2_2 = NestedUp(filters[2], filters[1], n_concat=3)
        self.up3_3 = NestedUp(filters[3], filters[2], n_concat=3)

        # Column 3
        self.up1_2 = NestedUp(filters[1], filters[0], n_concat=4)
        self.up2_3 = NestedUp(filters[2], filters[1], n_concat=4)

        # Column 4
        self.up1_3 = NestedUp(filters[1], filters[0], n_concat=5)

        # Deep supervision 출력 레이어
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Column 0 (인코더)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        # Column 1
        x0_1 = self.up1_0(x1_0, x0_0)
        x1_1 = self.up2_1(x2_0, x1_0)
        x2_1 = self.up3_2(x3_0, x2_0)
        x3_1 = self.up4_3(x4_0, x3_0)

        # Column 2
        x0_2 = self.up1_1(x1_1, x0_0, x0_1)
        x1_2 = self.up2_2(x2_1, x1_0, x1_1)
        x2_2 = self.up3_3(x3_1, x2_0, x2_1)

        # Column 3
        x0_3 = self.up1_2(x1_2, x0_0, x0_1, x0_2)
        x1_3 = self.up2_3(x2_2, x1_0, x1_1, x1_2)

        # Column 4
        x0_4 = self.up1_3(x1_3, x0_0, x0_1, x0_2, x0_3)

        # 출력
        if self.deep_supervision:
            # Deep supervision: 여러 출력의 평균 사용
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            # Deep supervision loss를 위해 모든 출력 반환
            return [output1, output2, output3, output4]
        else:
            # 최종 출력만 사용
            output = self.final(x0_4)
            return output
```

U-Net++ 추론

![result_000000.png](pics/result_000000.png)

U-Net++ 모델을 구현 후 세그멘테이션 결과까지 제출 하였다.

### **UNet·UNet++ 성능 비교 분석 결과**

두 모델 모두 도로 내부 영역은 충분히 정확하게 재현하여 **mIoU는 거의 동일한 수준(약 0.93)** 으로 수렴하였다.

UNet은 mIoU에서 근소한 우위를 보였고,

UNet++는 Boundary IoU에서 소폭 더 높은 성능을 보였다.

전체적으로 두 모델의 성능 차이는 **1%p 미만의 미세한 수준**이며,

내부 영역 중심의 지표(mIoU)는 UNet이, 경계 중심 지표(Boundary IoU)는 UNet++가 약하게 우세한 **구조적 경향성**을 확인하였다.

UNet (3 runs)

- mIoU 평균: 0.9315
- Boundary IoU 평균: 0.7185

UNet++ (3 runs)

- mIoU 평균: 0.9292
- Boundary IoU 평균: 0.7258

실험 A: Adam, 30 epoch, class = 2

| Model  | mIoU   | B-IoU  |
| ------ | ------ | ------ |
| UNet   | 0.8893 | 0.6402 |
| UNet++ | 0.9138 | 0.6934 |

앞선 실험과 달리, Adam, 30 epoch, class = 2에서 모든 지표에서 앞서는 경향을 볼 수 있다.

실험 B: Adam, 100 epoch, class = 34, batch size = 16

![training_comparison.png](pics/training_comparison.png)

### UNet / UNet++ 검증 지표 비교

| Model  | Best Epoch | Val mIoU | Val Boundary IoU |
| ------ | ---------- | -------- | ---------------- |
| UNet   | 100        | 0.2227   | 0.1846           |
| UNet++ | 87         | 0.2270   | 0.2018           |

### UNet / UNet++ 추론 사진 비교

UNet

![Figure_2.png](pics/Figure_2.png)

![unet.png](pics/unet.png)

UNet++

![Figure_1.png](pics/Figure_1.png)

![unetpp.png](pics/unetpp.png)

![comparison_6.png](pics/comparison_6.png)

![comparison_7.png](pics/comparison_7.png)

![comparison_8.png](pics/comparison_8.png)

![comparison_9.png](pics/comparison_9.png)

관찰 결과

### 1. **경계 표현력**

- **UNet++가 더 깨끗함**
  차량의 외곽선, 신호등 기둥 주변, 도로-잔디 경계에서 UNet보다 덜 번짐.
- **UNet 출력은 경계가 퍼짐**
  특히 도로 가장자리, 나무 밑 부분에서 흐림 존재.

### 2. **객체 내부 일관성**

- **UNet++ 차량 내부 색(클래스)가 더 일정함**
  차량 표면에 섞여 들어가는 오분류 픽셀이 줄어듦.
- **UNet → 차량 내부에 잡색 픽셀 혼입**
  물체 내부가 안정적으로 채워지지 않음.

### 3. **작은 물체 처리**

- **UNet++가 작은 객체(신호등, 표지판)를 더 잘 구분**
  - 신호등 위 녹색/파랑 부분의 segmentation 품질이 향상됨.
- **UNet은 작은 물체 주변이 더 흐릿함**

### 4. **도로 표면/잔디 영역 분리**

- **UNet++가 훨씬 안정적**
  도로 경계가 일정하고 유지됨.
- **UNet은 도로 끝부분에 잡음이 많음**

### 5. **배경(나무/하늘) 처리**

- **UNet++ → 배경 채우기 더 부드럽고 연속적**
- **UNet → 경계 주변 색 떨림(jitter)**

결론 : 클래스를 복잡하게 하거나, 특정 옵티마이저를 사용하였을 때, **UNet++이 UNet 보다**

Val mIoU, Val Boundary IoU에서 우월함을 알 수 있었다. 또한, 클래스를 단순하게 하거나 특정 옵티마이저를 사용하였을 때, Val mIoU는 비슷함을 Val Boundary IoU에서는 항상 우월함을 알 수 있었다.

두 모델 모두 동일한 구조적 출력이지만,

**UNet++는 Dense skip connection 덕분에 UNet보다 세밀한 영역에서 더 안정적인 분할을 수행**한다.

특히**차량 경계, 신호등/표지판 같은 작은 물체**,**도로 경계**,**객체 내부 일관성**에서 차이가 뚜렷하다.

U-Net과 U-Net++ 두 모델의 성능이 정량적/정성적으로 비교하였고,

U-Net++ 의 세그멘테이션 결과 사진과 IoU 계산치를 U-Net과 비교하였다.
