import torch.nn as nn
from ModificationLayer import *
import copy

class ResNet18( nn.Module ):
    def __init__(self, residualBlock, num_of_output_classes=1000, use_batchnorm=True,
                feature_layer_dim_before_FC=None, activation=nn.ReLU( inplace=True ), dropout_rate=0.0 ):
        """
        建構 ResNet 模型 ( 包含多個 ResidualBlock )
        預想空間尺寸解析度為 128*128

        Args:
            residualBlock (nn.Module): 傳入的已經打包好的 Residual Block
            num_of_output_classes (int): 輸出類別的數量, 默認為 1000
            use_batchnorm (bool): 是否使用批次正規化, 默認為 True
            feature_layer_dim (int or None): 指定特徵 avgpool 與 FC 之間的「特徵維度」, 用以先行一步降低特徵數防止過擬合, 默認為 None
            activation (nn.Module): 卷積後的激活函數, 默認為 nn.ReLU()
            dropout (float): 丟棄部份神經元的比例, 介於 0.0 - 1.0 之間, 默認為 0.0
        """
        
        super(ResNet18,self).__init__()

        # 初始化並賦值必要參數
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.feature_layer_dim_before_FC = feature_layer_dim_before_FC
        # 定義標準 ResNet 的每個 Block 應有的 ResidualBlock 數量和其輸出特徵數
        num_features = [64, 128, 256, 512]
        num_sections = [2, 2, 2, 2]
        # 第一個卷積層 ( 原 ResNet 論文第一層就是 7*7 的 kernal, 且 stride = 2、padding = 3 )
        self.conv1 = conv( 3 , num_features[0] , 7 , 2 , 3, activation , use_batchnorm = use_batchnorm , bias = False  )
        # 捲積層後的 maxpooing 層 ( 原 ResNet 論文 kernal = 3、stride = 2、padding = 1 )
        self.maxpool = nn.MaxPool2d( 3 , 2 , 1 )
        
        ############### 這邊開始疊 ResNet18 的 4 個階段共計 16 層 NN ###############
        sections = []
        # 依照 num_blocks 完成 4 個階段的封裝
        for idx in range( len( num_sections )-1 ):
            # 這邊 stride 要設定 1 還是 2 要測試, 原論文的結構其實不全是 1 或 2
            sections.append( self._build_blocks( residualBlock , num_features[idx] , num_features[idx+1] , 1 , num_sections[idx] ) ) 
        # 將封裝完的 4 階段變成模型序列
        self.sections = nn.Sequential( *sections )

        # 全連接層之前的「全局平均池化層」, 轉換維度成 512*1*1 以便後續的全連接層操作
        self.avgpool = nn.AdaptiveAvgPool2d( (1, 1) )
        
        # 如果有指定先行一步降低特徵維度的話
        if feature_layer_dim_before_FC is not None:
            self.FC0 = linear( num_features[-1] , feature_layer_dim_before_FC , use_batchnorm = use_batchnorm)
        
        # 依照 dropout_rate 進行部份神經元丟棄 ( 丟棄: 設置為 0 但不改變神經元總量與張量維度 )
        self.dropout = nn.Dropout( dropout_rate )
        # 最終 FC 輸出層
        FC_input_dim = feature_layer_dim_before_FC if feature_layer_dim_before_FC is not None else num_features[-1]
        self.FC = linear( FC_input_dim , num_of_output_classes , use_batchnorm = False )

    def _build_blocks( self, residualBlock, in_channels, out_channels, stride, num_of_residual_block ):
        """
        構建「多階段 ( 一般為 4 階段 )」殘差塊
        一個 Residual Block 包含 2 層 NN

        Args:
            residualBlock (nn.Module): 傳入的殘差塊 ( 已包含 2 層 NN 或 bottleneck 3 層 NN )
            in_channels (int): ResidualBlock 內部的輸入通道數
            out_channels (int): ResidualBlock 內部的輸出通道數
            stride (int): 用於控制 ResidualBlock 內部的 NN 層的 stride
            num_of_residual_block (int): 階段內的 Residual Block 數量。

        Returns:
            nn.Sequential: 序列化的殘差塊。
        """
        layers = []
        # 依照 num_of_residual_block 參數
        for i in range( num_of_residual_block ):
            singleResidualBlock = residualBlock( in_channels , out_channels , stride , activation = copy.deepcopy(self.activation) , use_batchnorm = self.use_batchnorm )
            layers.append( singleResidualBlock )

        return nn.Sequential( *layers )

    def forward(self, x, use_dropout=False):
        """
        Args:
            x (Tensor): 3*128*128 張量
            use_dropout (bool): 是否丟棄部份神經元, 默認為 False

        Returns:
            out (Tensor): (batch_size, num_of_output_classes) 的輸出類別張量
            out_FC0 (Tensor or None): 若有定義 FC0, 則為 FC0 的輸出特徵, 否則為 None
        """

        # 通過第一個卷積層和最大池化層
        x = self.conv1(x)
        x = self.maxpool(x)

        # 通過所有殘差塊
        x = self.sections(x)

        # 通過全局平均池化層
        x = self.avgpool(x)
        
        # 攤平操作, 得到 ( batch_size, 512 )
        x = x.view(x.size(0), -1)

        # 初始化 FC0 輸出變數
        out_FC0 = None

        # 如果定義了「提前特徵降維」層
        if hasattr(self, 'FC0'):
            x = self.FC0(x)
            out_FC0 = x

        # 根據 use_dropout 來應用 Dropout
        if use_dropout:
            x = self.dropout(x)

        # 最終的全連接層
        out = self.FC(x)

        return out, out_FC0

    def resnet18( fm_mult = 1.0 , **kwargs ):
        num_features = [64,128,256,512]
        for i in range(len(num_features)):
            num_features[i] = int( num_features[i] * fm_mult )
        model = ResNet18(ResidualBlock, [2,2,2,2] , num_features , **kwargs ) 
        return model