#borrowed from https://github.com/tonylins/pytorch-mobilenet-v2

import torch.nn as nn
import math

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=6):
        """
        定義「反殘差」Block, 由
        1. 1x1 kernal 的 expand layer
        2. 3x3 kernal 的 depthwise layer
        3. 1x1 kernal 的 project layer

        Args:
            inp (int): 輸入通道數
            oup (int): 輸出通道數
            stride (int): 步幅, 默認為 1
            expand_ratio (int): 擴展係數, 用於擴展通道數, 默認為 6 倍
        """
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        # 判斷是否使用快捷連接, 原論文的條件: 必須在 stride 為 1 且輸出通道等於輸入通道時才有, 反之沒有捷徑
        self.use_res_connect = self.stride == 1 and inp == oup

        # 組合 Block 的內容
        self.conv = nn.Sequential(
            # 1x1 kernal 的 expand layer
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # 3x3 kernal 的 depthwise layer
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # 1x1 kernal 的 project layer
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # 定義 MobileNetV2 的 Inverted Residual Block 數量以及各 Block 的 stide、通道放大倍率
        
        """
        t -> expansion factor: 通道放大倍率
        c -> output channels : 輸出通道數
        n -> number of repeat: 重複次數, 表示這樣定義的 Inverted Residual Block 共要有幾個
        s -> stride          : 捲積的步長, 1 為不改變輸出空間尺寸, 2 的話特徵圖的寬合高會被減半
        """
        self.interverted_residual_setting = [
            # t,  c,  n, s, 這邊為 MobileNetV2 原始論文的架構
            [ 1,  16, 1, 1],
            [ 6,  24, 2, 2],
            [ 6,  32, 3, 2],
            [ 6,  64, 4, 2],
            [ 6,  96, 3, 1],
            [ 6, 160, 3, 2],
            [ 6, 320, 1, 1],
        ]

        # assert input_size % 16 == 0
        # 建構第一層
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        ##### 依照設置, 建構所有 Inverted Residual Block #####
        input_channel = 32
        self.bottlenecks = nn.ModuleList()
        for expansion_factor, output_channel, number_of_repeat, stride in self.interverted_residual_setting:
            # 根據 number_of_repeat 多次建構同樣的 Inverted Residual Block
            for idx in range( number_of_repeat ):
                if idx == 0:
                    self.bottlenecks.append(InvertedResidual(input_channel, output_channel, stride, expansion_factor))
                else:
                    self.bottlenecks.append(InvertedResidual(input_channel, output_channel, 1, expansion_factor))
                # 更新下一次的 input dim 為此層的 output dim
                input_channel = output_channel

        # 建構最後一層
        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        # 池化層
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Dropout 層 ( 原始論文為 0.2 )
        self.dropout = nn.Dropout( p=0.2 )
        # FC 層, 每個部位的輸出層
        self.head_eyes_left = nn.Linear(1280, 2)  # 輸出格式為 x, y
        self.head_eyes_right = nn.Linear(1280, 2)
        self.head_mouth = nn.Linear(1280, 2)
        self.head_nose = nn.Linear(1280, 2)

        self._initialize_weights()

    def forward(self, x , use_dropout=False):
        # 第一層
        x = self.conv1(x)
        # 7 個 Bottleneck 共計 17 層 Inverted Residual Block
        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
        # 最後一層
        x = self.conv2(x)
        # 全局池化層
        x = self.avgpool(x)
        # 展平
        x =  x.view(x.size(0), -1) 
        # Dropout
        if use_dropout:
            x = self.dropout(x)
        # 全連接層進行特徵映射, 分別獲得每個部位的座標預測
        eyes_left = self.head_eyes_left(x)
        eyes_right = self.head_eyes_right(x)
        nose = self.head_nose(x)
        mouth = self.head_mouth(x)

        return eyes_left, eyes_right, nose, mouth

    def _initialize_weights(self):
        # 遍歷模型中的所有模塊
        for m in self.modules():

            # 判斷是否為卷積層 (Conv2d)
            if isinstance(m, nn.Conv2d):
                # 計算權重初始化的標準差參數 n
                # n 是根據 He 初始化方法設定, 為卷積核的元素數目 ( 寬 * 高 * 輸出通道數 )
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 權重使用 He 初始化方法，以 0 為均值，sqrt(2. / n) 為標準差的正態分布初始化
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # 如果卷積層具有偏置項，則將其初始化為 0
                if m.bias is not None:
                    m.bias.data.zero_()

            # 判斷是否為批次標準化層 ( BatchNorm2d )
            elif isinstance(m, nn.BatchNorm2d):
                # 將批次標準化層的權重全部初始化為 1
                m.weight.data.fill_(1)
                # 將批次標準化層的偏置全部初始化為 0
                m.bias.data.zero_()
            
            # 判斷是否為全連接層 ( Linear )
            elif isinstance(m, nn.Linear):
                # 計算全連接層權重初始化的標準差參數 n
                # n 是全連接層的輸入特徵數量
                n = m.weight.size(1)
                # 全連接層權重使用較小的標準差 0.01 的正態分布初始化
                m.weight.data.normal_(0, 0.01)
                # 全連接層偏置初始化為 0
                m.bias.data.zero_()