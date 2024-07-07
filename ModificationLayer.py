#wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal , kaiming_normal

def sequential(*kargs ):
    """
    建立一個序列化的模型層，並自動設定輸出通道數。

    Args:
        *kargs (nn.Module): 可變數量的 PyTorch 模組，這些模組將會被按順序疊加在一起。

    Returns:
        nn.Sequential: 返回一個包含所有輸入模組的序列化模組，並附帶自動設定的輸出通道屬性。
    """
    seq = nn.Sequential(*kargs)
    # 從最後一層往前抓取可能的「特徵」或「通道」輸出，並擴展 Sequential 一個自定義變數 out_channels 或 out_features
    for layer in reversed(kargs):
        if hasattr( layer , 'out_channels'):
            seq.out_channels = layer.out_channels
            break
        if hasattr( layer , 'out_features'):
            seq.out_channels = layer.out_features
            break
    return seq

def weight_initialization( weight , init , activation):
    """
    初始化神經網絡層的權重
    支持 Kaiming 初始化和 Xavier 初始化。
        - kaiming: 適用於 ReLU 和其變體
        - xavier:  適用於 Sigmoid 或 Tanh

    Args:
        weight (Tensor): 欲初始化的權重張量。
        init (str): 指定初始化方法的字符串，可以是 'kaiming' 或 'xavier'。
        activation (nn.Module): 與權重相關聯的激活函數，用於 Kaiming 初始化時提供額外參數。

    Returns:
        None: 此函數不返回任何值，但會直接修改傳入的權重張量。
    """
    if init is None:
        return
    
    if init == "kaiming":
        if hasattr(activation, "negative_slope"):
            # kaiming 初始化的 a 參數為 負斜率 ( LeakyReLU 為負，ReLU 為 0 )
            kaiming_normal( weight , a = activation.negative_slope )
        else:
            kaiming_normal( weight , a = 0 )
    elif init == "xavier":
        xavier_normal( weight )
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding = 0 , init = "kaiming" , activation = nn.ReLU() , use_batchnorm = False , pre_activation = False ):
    """
    創建一個卷積層，但這捲積層可以直接根據參數完成「激勵層」、「批次正規化層」的序列化包裝
    可選參數包括
        - 步長 ( stride )
        - 填充 ( padding )
        - 支持權重初始化 ( init )
        - 激活函數 ( activation )
        - 批次正規化 ( use_batchnorm )

    Args:
        in_channels (int): 卷積層的輸入通道數
        out_channels (int): 卷積層的輸出通道數
        kernel_size (int or tuple): 卷積核的大小

        stride (int or tuple): 卷積的步長, 默認為 1
        padding (int or tuple): 應用於輸入的填充量, 默認為 0
        init (str): 權重初始化方法, 支持 "kaiming" 或 "xavier", 默認為 "kaiming"
        activation (nn.Module): 卷積後的激活函數, 默認為 nn.ReLU()
        use_batchnorm (bool): 是否在卷積層和激活函數之間添加批次正規化層, 默認為 False
        pre_activation (bool): 是否將正規化層與激勵層移至捲積層之前? 默認為 False, 通常只有「較深」的網路會用到 True

    Returns:
        nn.Sequential: 包含卷積層, 可選的批次正規化層和激活函數的序列化模組。
    """
    
    layers = []
    
    # 檢查 padding 是否為 list, 如果是的話要進一步處理
    if type( padding ) == type( list() ) :
        """
        padding 參數的值可以是
            - 1 個: 4 個邊 ( 左、右、上、下 ) 為同一數值
            - 2 個: 區分為「左右」與「上下」共用一個數值
            - 4 個: 4 個邊各自為一個數值
        因此這邊 padding 參數的數量不能為 3, 且 4 個的時候要特別處理
        """
        assert len( padding ) != 3 
        if len( padding ) == 4:
            layers.append( nn.ReflectionPad2d( padding ) )
            # 因為 pytorch 的 4 個邊使用不同的數量填充, 需要由 nn.ReflectionPad2d 來完成
            # 由於做過填充, 因此將 padding 設為 0
            padding = 0
    # 根據 batchnorm 來判斷是否要增加 bias, 因為 batchnorm 中其實就有 bias 的功能
    bias = not use_batchnorm

    # 定義捲積層
    conv_layer = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding , bias = bias)
    # 權重初始化
    weight_initialization( conv_layer , init , activation )
    # 將捲積層加入 layers 中
    layers.append( conv_layer )
    
    """
    其實這邊「正規化層」和「激勵層」誰先誰後是一個...小問題
        - ( Conv -> Normalization -> Activation ): 
            1. 批量正規化可以控制激勵層的輸入範圍, 有助於避免激勵層 ( 如ReLU ) 的激活值過大或過小的問題, 有助於提升穩定性
            2. 批量正規化在卷積後直接使用可以有效減少所謂的內部協方差偏移 ( Internal Covariate Shift ), 這可以讓收斂過程更穩定和快速
        - ( Conv -> Activation -> Normalization ):
            1. 使用特定類型的激勵函數時, 例如對於具有飽和性質的激勵函數 ( 如 Sigmoid 或 Tanh ) 時, 可能希望先讓網絡學到非線性特徵再進行正規化
    """
    # 如果 pre_activation 為 True, 則在捲積層之前先加入正規化層和激勵層, 反之加在捲積層之後
    if pre_activation:
        layers = _batchnorm_and_activation_layer( in_channels, activation, use_batchnorm ) + layers
    else:
        layers += _batchnorm_and_activation_layer( out_channels, activation, use_batchnorm )
    
    seq = nn.Sequential( *layers )
    seq.out_channels = out_channels
    return seq

def _batchnorm_and_activation_layer(specific_channels, activation, use_batchnorm):
    """
    根據激勵函數的類型決定正規化層和激勵層的先後順序，並返回這些層的列表。

    Args:
        specific_channels (int): 正規化層需要正規化的通道數
        activation (nn.Module): 激勵函數
        use_batchnorm (bool): 是否使用批次正規化

    Returns:
        return_layers (list): 根據需要, 包含正規化層和激勵層的 list
    """

    return_layers = []

    # 定義非線性的激勵函數
    nonlinear_activations = (nn.Sigmoid, nn.Tanh)
    
    # 先判斷需不需要正規化層
    if use_batchnorm:
        # 再依照激勵函數的種類來判斷正規化層和激勵層哪個先
        if isinstance( activation(), nonlinear_activations ):
            return_layers.append( activation )
            return_layers.append( nn.BatchNorm2d( specific_channels ) )
        else:
            return_layers.append( nn.BatchNorm2d( specific_channels ) )
            return_layers.append( activation )
    # 如果不需要正規化層, 就僅需要增加激勵層
    else:
        return_layers.append(activation)

    return return_layers
    
def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , init = "kaiming" , activation = nn.ReLU() , use_batchnorm = False, pre_activation = False ):
    """
    創建一個「反」卷積層 ( 轉置卷積 ), 可選參數包括
        - 步長 ( stride )
        - 填充 ( padding )
        - 輸出填充 ( output_padding )
        - 支持權重初始化 ( init )
        - 激活函數 ( activation )
        - 批次正規化 ( use_batchnorm )

    Args:
        in_channels (int): 反卷積層的輸入通道數
        out_channels (int): 反卷積層的輸出通道數
        kernel_size (int or tuple): 卷積核的大小

        stride (int or tuple): 反卷積的步長, 默認為 1 ( 不可為 0 )
        padding (int or tuple): 用於擴增輸入圖層後的邊緣刪除量, 默認為 0
        output_padding (int or tuple): 反卷積操作最終結果的額外填充, 用於控制輸出的大小, 默認為 0
        init (str): 權重初始化方法, 支持 "kaiming" 或 "xavier", 默認為 "kaiming"
        activation (nn.Module): 反卷積後的激活函數, 默認為 nn.ReLU()
        use_batchnorm (bool): 是否在反卷積層和激活函數之間添加批次正規化層, 默認為 False
        pre_activation (bool): 是否將正規化層與激勵層移至捲積層之前? 默認為 False, 通常只有「較深」的網路會用到 True

    Returns:
        nn.Sequential: 包含反卷積層，可選的批次正規化層和激活函數的序列化模組。
    """
    
    layers = []
    # 根據 batchnorm 來判斷是否要增加 bias, 因為 batchnorm 中其實就有 bias 的功能
    bias = not use_batchnorm
    # 定義捲積層
    deconv_layer = nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding , bias = bias )
    # 權重初始化
    weight_initialization( deconv_layer , init , activation )
    # 新增反捲積層
    layers.append( deconv_layer )

    if pre_activation:
        layers = _batchnorm_and_activation_layer( in_channels, activation, use_batchnorm ) + layers
    else:
        layers += _batchnorm_and_activation_layer( out_channels, activation, use_batchnorm )
    
    seq = nn.Sequential( *layers )
    seq.out_channels = out_channels
    return seq

def linear( in_channels , out_channels , activation = None , use_batchnorm = False):
    """
    此為線性層 ( 相當於全連接層 )
    
    Args:
        in_channels (int): 輸入的通道數
        out_channels (int): 輸出的通道數

        activation (torch.nn.Module): 激活函數, 默認為 None -> 不使用激活函數。
        use_batchnorm (bool): 是否添加批次正規化層, 默認為 False

    Returns:
        torch.nn.Sequential: 包含線性層，可選的批量正規化層和激活函數的序列化模組。
    """

    layers = []
    # 根據 batchnorm 來判斷是否要增加 bias, 因為 batchnorm 中其實就有 bias 的功能
    bias = not use_batchnorm

    layers.append( nn.Linear( in_channels , out_channels , bias = bias ) )
    # 正規化層
    if use_batchnorm:
        layers.append( nn.BatchNorm1d( out_channels ))
    # 激勵層
    if activation is not None:
        layers.append( activation )

    return nn.Sequential( *layers )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels , 
                out_channels = None, 
                kernel_size = 3, 
                stride = 1,
                padding = None , 
                weight_init = "kaiming" , 
                activation = nn.ReLU() ,
                is_bottleneck = False ,
                use_projection = False,
                scaling_factor = 1.0,
                is_inplace_of_activation = False,
                use_batchnorm = False
                ):
        """
        初始化殘差塊，支持標準和瓶頸結構。

        Args:
            in_channels (int): 輸入通道數

            out_channels (int): 輸出通道數, 如果為 None 則默認為 ( in_channels // stride ), 默認為 None
            kernel_size (int or tuple): 卷積核大小, 默認為 3*3
            stride (int or tuple): 卷積步長, 默認為 1
            padding (int or tuple): 填充大小, 如果為 None 則默認為 ( kernel_size - 1) // 2, 默認為 None
            weight_init (str): 權重初始化方法, 支持 "kaiming" 或 "xavier", 默認為 "kaiming"
            activation (nn.Module): 激活函數, 默認為 nn.ReLU( inplace = is_inplace_of_activation )
            is_bottleneck (bool): 是否使用瓶頸結構, 默認為 False
            use_projection (bool): 是否使用投影快捷連接, 默認為 False
            scaling_factor (float): 殘差塊輸出的縮放因子, 用於調整快捷連接的影響程度, 如設置 0 表示快捷連接不存在, 默認為 1.0
            is_inplace_of_activation (bool): 是否讓 激勵函數 直接操作記憶體, 默認為 False
            use_batchnorm (bool): 是否添加批次正規化層, 默認為 False
        """
        super(type(self),self).__init__()

        ############### 為基本參數賦值 ###############
        # 如果沒有輸出通道數, 則使用默認
        self.out_channels = in_channels // stride if out_channels is None else out_channels
        # 如果沒有 padding, 就判斷是否是特徵提取用 ( is_inplace_of_activation 為 True ), 如果是就指定 padding 為 1, 反之使用默認
        self.padding = ( 1 if is_inplace_of_activation else (kernel_size - 1)//2 ) if padding is None else padding
        # 判斷 is_inplace_of_activation 和 激勵函數是否符合條件, 否則使用傳入的激勵函數 ( 包含 None: 不使用 )
        self.activation = nn.ReLU( inplace = is_inplace_of_activation ) if is_inplace_of_activation and isinstance( activation(), nn.ReLU() ) else activation
        self.use_projection = use_projection
        self.scaling_factor = scaling_factor

        convs = []
        # 重新定義快捷投影選項:
        #   - 如果本身為 True, 則為 True
        #   - 如果本身為 False 但 stride 不為 1 ( 會降低空間尺寸 ) 或輸出維度不等於輸入維度時
        self.use_projection = use_projection or (stride != 1 or in_channels != out_channels)
        # 如果快捷投影為 True, 使用 1x1 卷積進行輸出的空間尺寸匹配, 反之使用 恆等映射
        self.shortcut = conv( in_channels, out_channels, 1, stride, 0, weight_init, None, False) if use_projection else nn.Sequential()

        # 如果使用「瓶頸結構」
        if is_bottleneck:
            convs.append( conv(      in_channels     ,      in_channels//2  ,           1 ,      1 ,                    0 , weight_init , self.activation , use_batchnorm, False ))
            convs.append( conv(      in_channels//2  , self.out_channels//2 , kernel_size , stride , (kernel_size - 1)//2 , weight_init , self.activation , use_batchnorm, False ))
            # 最後一層不需要權重初始化和激勵函數層
            convs.append( conv( self.out_channels//2 ,    self.out_channels ,              1 ,      1 ,                 0 ,        None ,            None , use_batchnorm, False )) 
        else:
            convs.append( conv(          in_channels ,          in_channels ,    kernel_size ,      1 ,      self.padding , weight_init , self.activation , use_batchnorm, False ))
            # 最後一層不需要權重初始化和激勵函數層
            convs.append( conv(          in_channels ,    self.out_channels ,    kernel_size ,      1 ,      self.padding ,        None ,            None , use_batchnorm, False ))
        
        
        self.layers = nn.Sequential( *convs )
    def forward(self, x):
        # 殘差連接 = 主路徑 ( layers ) + 快捷連接縮放因子 * 快捷連接
        out = self.layers(x) + self.scaling_factor * self.shortcut(x)

        return self.activation( out )