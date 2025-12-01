import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) x 2 구조의 블록"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool로 다운샘플링 후 double conv 수행"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """업샘플링 후 double conv 수행"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # bilinear를 사용하면 일반 convolution으로 채널 수를 줄임
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 입력 형태는 CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    참고 사이트
    https://arxiv.org/abs/1505.04597
    https://github.com/ZJUGiveLab/UNet-Version
    
    기본 U-Net 아키텍처로 의료 영상 segmentation을 위해 고안됨
    """
    def __init__(self, in_channels=3, num_classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 인코더 (다운샘플링 경로)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 디코더 (업샘플링 경로)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 출력 레이어
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 인코더
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 디코더
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 출력 (sigmoid/softmax 없음, CrossEntropyLoss 사용)
        logits = self.outc(x)
        return logits


class NestedUp(nn.Module):
    """U-Net++를 위한 다중 skip connection이 있는 업샘플링 블록"""
    def __init__(self, in_channels, out_channels, n_concat=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 * n_concat, out_channels)

    def forward(self, *inputs):
        """
        inputs: 연결할 텐서들의 리스트
        inputs[0]: 하위 레벨에서 온 텐서 (업샘플링 대상)
        inputs[1:]: skip connection 텐서들
        """
        x = self.up(inputs[0])
        
        # 모든 skip connection과 연결
        skip_connections = list(inputs[1:])
        x = torch.cat([x] + skip_connections, dim=1)
        
        return self.conv(x)


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


if __name__ == '__main__':
    # U-Net 테스트
    print("=" * 50)
    print("U-Net 테스트")
    print("=" * 50)
    unet = UNet(in_channels=3, num_classes=2)
    x = torch.randn(1, 3, 256, 768)
    output = unet(x)
    print(f"입력 크기: {x.shape}")
    print(f"출력 크기: {output.shape}")
    print(f"총 파라미터 수: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f}M")
    
    print("\n" + "=" * 50)
    print("U-Net++ 테스트")
    print("=" * 50)
    unetpp = UNetPlusPlus(in_channels=3, num_classes=2, deep_supervision=False)
    output = unetpp(x)
    print(f"입력 크기: {x.shape}")
    print(f"출력 크기: {output.shape}")
    print(f"총 파라미터 수: {sum(p.numel() for p in unetpp.parameters()) / 1e6:.2f}M")
    
    print("\n" + "=" * 50)
    print("U-Net++ (Deep Supervision) 테스트")
    print("=" * 50)
    unetpp_ds = UNetPlusPlus(in_channels=3, num_classes=2, deep_supervision=True)
    outputs = unetpp_ds(x)
    print(f"입력 크기: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"출력 {i+1} 크기: {out.shape}")
    print(f"총 파라미터 수: {sum(p.numel() for p in unetpp_ds.parameters()) / 1e6:.2f}M")
