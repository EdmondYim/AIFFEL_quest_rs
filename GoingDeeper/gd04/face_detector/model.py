import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1, 1)):
        super(DepthwiseConvBlock, self).__init__()
        self.strides = strides

        if strides != (1, 1):
            self.pad = nn.ZeroPad2d((1, 1, 1, 1))
        else:
            self.pad = nn.Identity()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=strides, padding=0 if strides != (1, 1) else 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.depthwise(x)
        x = F.relu(self.bn1(x))
        x = self.pointwise(x)
        return F.relu(self.bn2(x))

class BranchBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(BranchBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, filters * 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x1 = self.conv2(x1)
        x2 = self.conv3(x)
        x = torch.cat([x1, x2], dim=1)
        return F.relu(x)

class HeadBlock(nn.Module):
    def __init__(self, in_channels, num_cell, out_channels):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_cell * out_channels, kernel_size=3, stride=1, padding=1)
        self.num_cell = num_cell
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, self.out_channels)
        return out

class SSD(nn.Module):
    def __init__(self, num_classes=2, input_shape=(3, 256, 320), num_cells=[3, 2, 2, 3]):
        super(SSD, self).__init__()

        self.base_channel = 16
        self.num_cells = num_cells
        self.num_classes = num_classes

        self.conv_blocks = nn.ModuleList([
            DepthwiseConvBlock(3, self.base_channel * 4, strides=(1, 1)),
            BranchBlock(self.base_channel * 4, self.base_channel * 8)
        ])

        self.layers = nn.ModuleList([
            DepthwiseConvBlock(self.base_channel * 8, self.base_channel * 16, strides=(2, 2)),
            BranchBlock(self.base_channel * 16, self.base_channel),
            DepthwiseConvBlock(self.base_channel * 16, self.base_channel * 16, strides=(2, 2)),
            BranchBlock(self.base_channel * 16, self.base_channel)
        ])
        
# 다중 스케일(feature map) 생성을 위한 추가 레이어들
# 최종 출력 feature map의 stride가 8, 16, 32, 64가 되도록 만들어야 한다.

# 노트북에 있던 기존 SsdModel 기준으로 보면:
# x1 = conv_blocks[0](x) -> stride 1
# x2 = conv_blocks[1](x1) -> stride 1
# x3 = layers[0](x2) -> stride 2
# x3 = layers[1](x3) -> stride 2
# x4 = layers[2](x3) -> stride 4
# x4 = layers[3](x4) -> stride 4

# 그런데 노트북의 SsdModel이 out하는 extra_layers 는 [x1, x2, x3, x4] 이고,
# 이것들의 stride는 1, 1, 2, 4에 해당한다.
# 하지만 default_box는 stride 8, 16, 32, 64를 요구한다.
# 둘 사이에 큰 불일치가 존재한다.

# 노트북의 입력 크기를 다시 확인해보자.
# IMAGE_WIDTH = 320, IMAGE_HEIGHT = 256
# stride가 1, 1, 2, 4라면 feature map 크기는
# 320x256, 320x256, 160x128, 80x64가 된다.
# 반면 default_box에서 기대하는 feature map 크기는
# 320/8=40, 320/16=20, 320/32=10, 320/64=5 이다.

# 즉, 노트북의 기존 SsdModel은 default_box와 구조가 맞지 않는다.
# 그대로 사용할 경우 shape mismatch(형상 불일치) 오류가 발생한다.

# 따라서 SsdModel을 반드시 수정하여
# stride 8, 16, 32, 64에 해당하는 feature map이 나오도록 만들어야 한다.
# 이를 위해 downsampling 레이어를 더 추가하겠다.

# stride 8, 16, 32, 64를 만족하도록 수정한 SsdModel의 구조:
# 입력: 320x256
# Block 1: -> /2
# Block 2: -> /4
# Block 3: -> /8  (Feature 1)
# Block 4: -> /16 (Feature 2)
# Block 5: -> /32 (Feature 3)
# Block 6: -> /64 (Feature 4)

# DepthwiseConvBlock과 BranchBlock을 그대로 재사용하되,
# 이를 위 구조가 되도록 재배치한다.
        
        self.features = nn.ModuleList([
            # /2
            DepthwiseConvBlock(3, 32, strides=(2, 2)), 
            # /4
            DepthwiseConvBlock(32, 64, strides=(2, 2)),
            # /8 (Feature 1)
            DepthwiseConvBlock(64, 128, strides=(2, 2)),
            # /16 (Feature 2)
            DepthwiseConvBlock(128, 256, strides=(2, 2)),
            # /32 (Feature 3)
            DepthwiseConvBlock(256, 256, strides=(2, 2)),
            # /64 (Feature 4)
            DepthwiseConvBlock(256, 256, strides=(2, 2))
        ])
        
        self.conf_layers = nn.ModuleList()
        self.loc_layers = nn.ModuleList()
        
        in_channels_list = [128, 256, 256, 256]
        
        for i, num_cell in enumerate(num_cells):
            self.conf_layers.append(HeadBlock(in_channels_list[i], num_cell, self.num_classes))
            self.loc_layers.append(HeadBlock(in_channels_list[i], num_cell, 4))

        self._init_weights()

    def forward(self, x):
        # x: 3, 256, 320
        x = self.features[0](x) # /2
        x = self.features[1](x) # /4
        
        x1 = self.features[2](x) # /8
        x2 = self.features[3](x1) # /16
        x3 = self.features[4](x2) # /32
        x4 = self.features[5](x3) # /64
        
        features = [x1, x2, x3, x4]
        
        confs = []
        locs = []

        for i, feature in enumerate(features):
            confs.append(self.conf_layers[i](feature))
            locs.append(self.loc_layers[i](feature))

        confs = torch.cat(confs, dim=1)
        locs = torch.cat(locs, dim=1)

        return torch.cat([locs, confs], dim=2)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
