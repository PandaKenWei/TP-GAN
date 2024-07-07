"""
模型結構直接參考 https://arxiv.org/pdf/1704.04086

G 裡面包含的過程有
    - 4 個 LocalPathway model 的特徵提取結果
    - 將上一步的 4 個結果丟進 LocalFuser 

    ..... todo
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from ModificationLayer import *
from UtilityMethods import resize_tensor, elementwise_multiply_and_cast_to_int as EMaC2I

class LocalPathway(nn.Module):
    def __init__(self, use_batchnorm = True, feature_layer_dim = 64 , FM_multiplier = 1.0):
        """
        LocalPathway 模型, 用於提取和重建影像的局部特徵, 像是
            - 左眼
            - 右眼
            - 鼻子
            - 嘴吧
        四個部份, 然後這個模型內部結構包含 4 個編碼器與 2 個解碼器
        其中
            - 編碼器的輸出特徵圖分別為 64, 128, 256, 512
            - 解碼器的輸出特徵圖分別為 256, 128
        如果要因應計算量進行調整的話, 就調整 FM_multiplier 這個參數
        
        Args:
            use_batchnorm (bool): 是否使用批次正規化, 默認為 True
            feature_layer_dim (int): 最終特徵層的維度, 默認為 64
            FM_multiplier (float): feature map multiplier, 特徵圖數量的乘數, 用於調整每層的特徵圖數量, 默認為 1.0
                - 設置為 1.0, 則特徵圖數量保持不變
                - 設置為 0.5, 則每層的特徵圖數量減半, 減少計算量和記憶體開銷
                - 設置為 2.0, 則每層的特徵圖數量翻倍, 增加模型容量、總計算量並增加記憶體開銷
        """
        super(LocalPathway,self).__init__()

        # 設定編碼器和解碼器的數量, 以及每個編、解碼器輸出的特徵圖數量 ( FM: feature map, n: number )
        n_FM_encoder = [64, 128, 256, 512] # 4 個編碼器, list 的值為每個編碼器的輸出特徵圖數量
        n_FM_decoder = [256, 128]           # 2 個解碼器, list 的值為每個解碼器的輸出特徵圖數量
        n_FM_encoder = EMaC2I( n_FM_encoder, FM_multiplier )
        n_FM_decoder = EMaC2I( n_FM_decoder, FM_multiplier )

        ############### 4 個編碼器層, 每個編碼器層都為 1 個 NN 層和 1 個殘差連接塊 ###############
        # (batch_size, 3, H, W) -> (batch_size, n_fm_encoder[0], H, W)
        self.conv0 = sequential( conv(               3 , n_FM_encoder[0] , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                 ResidualBlock( n_FM_encoder[0] , activation = nn.LeakyReLU() ) )
        # 第二層開始 stride 變為 2, 逐層將空間維度減半
        # (batch_size, n_fm_encoder[0], H, W) -> (batch_size, n_fm_encoder[1], H//2, W//2)
        self.conv1 = sequential( conv( n_FM_encoder[0] , n_FM_encoder[1] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                 ResidualBlock( n_FM_encoder[1] , activation = nn.LeakyReLU() ) )
        # (batch_size, n_fm_encoder[1], H//2, W//2) -> (batch_size, n_fm_encoder[2], H//4, W//4)
        self.conv2 = sequential( conv( n_FM_encoder[1] , n_FM_encoder[2] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                 ResidualBlock( n_FM_encoder[2] , activation = nn.LeakyReLU() ) )
        # (batch_size, n_fm_encoder[2], H//4, W//4) -> (batch_size, n_fm_encoder[3], H//8, W//8)
        self.conv3 = sequential( conv( n_FM_encoder[2] , n_FM_encoder[3] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                 ResidualBlock( n_FM_encoder[3] , activation = nn.LeakyReLU() ) )
        
        ############### 2 個解碼器層, 每個解碼器層都為 1 個 DNN 層、1 個 NN 層和 1 個殘差連接塊 ###############
        # (batch_size, n_FM_encoder[3], H//8, W//8) -> (batch_size, n_FM_decoder[0], H//4, W//4)
        self.deconv0 =       deconv(                 n_FM_encoder[3] ,   n_FM_decoder[0] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm) 
        # (batch_size, n_FM_decoder[0] + n_FM_encoder[2], H//4, W//4) -> (batch_size, n_FM_decoder[0], H//4, W//4)
        self.after_select0 = sequential( conv(   n_FM_decoder[0] + self.conv2.out_channels ,    n_FM_decoder[0] , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm ) ,
                                         ResidualBlock(   n_FM_decoder[0] , activation = nn.LeakyReLU() ) )
        # (batch_size, n_fm_decoder[0], H//4, W//4) -> (batch_size, n_fm_decoder[1], H//2, W//2)
        self.deconv1 =       deconv( self.after_select0.out_channels ,   n_FM_decoder[1] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm) 
        # (batch_size, n_fm_decoder[1] + n_fm_encoder[1], H//2, W//2) -> (batch_size, n_fm_decoder[1], H//2, W//2)
        self.after_select1 = sequential( conv(   n_FM_decoder[1] + self.conv1.out_channels ,    n_FM_decoder[1] , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm ) ,
                                         ResidualBlock(   n_FM_decoder[1] , activation = nn.LeakyReLU() ) )
        # (batch_size, n_fm_decoder[1], H//2, W//2) -> (batch_size, feature_layer_dim, H, W)
        self.deconv2 =       deconv( self.after_select1.out_channels , feature_layer_dim , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm) 
        self.after_select2 = sequential( conv( feature_layer_dim + self.conv0.out_channels , feature_layer_dim  , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm ) ,
                                         ResidualBlock( feature_layer_dim , activation = nn.LeakyReLU() ) )
        
        # 最終的卷積層, 生成局部影像, 因為是最後一層, 不需要使用權重初始化和激勵函數
        # (batch_size, feature_layer_dim, H, W) -> (batch_size, 3, H, W)
        self.local_img = conv( feature_layer_dim , 3 , 1 , 1 , 0 , None , None , False )

                            
    def forward(self,x):
        """
        Args:
            x (torch.Tensor): 經過 Dlib 切割過的「部位」小圖, 形狀為 (batch_size, channels, height, width)
        
        Returns:
            local_img (torch.Tensor): 重建的局部圖像, 空間形狀與輸入相同
            deconv2 (torch.Tensor): 最後一層的特徵圖
        """
        ##### 編碼過程 #####
        conv0 = self.conv0( x )
        conv1 = self.conv1( conv0 )
        conv2 = self.conv2( conv1 )
        conv3 = self.conv3( conv2 )
        ##### 解碼過程 #####
        deconv0 = self.deconv0( conv3 )
        after_select0 = self.after_select0( torch.cat([deconv0,conv2],  1) ) # 需要以張量的 Channel 維度來與之前的層的特徵結果合併
        deconv1 = self.deconv1( after_select0 )
        after_select1 = self.after_select1( torch.cat([deconv1,conv1] , 1) )
        deconv2 = self.deconv2( after_select1 )
        after_select2 = self.after_select2( torch.cat([deconv2,conv0],  1 ) )
        # 產局部圖像
        local_img = self.local_img( after_select2 )
        # 確保輸出的影像尺寸與輸入一致
        assert local_img.shape == x.shape  ,  "{} {}".format(local_img.shape , x.shape)
        # 返回局部影像和最後一層的特徵
        return  local_img , deconv2

class LocalFuser(nn.Module):
    """
    LocalFuser 模型, 用於融合 4 個局部特徵 -> 左眼、右眼、鼻子和嘴巴
        
    處理流程：
        將每個局部特徵填充到指定的位置並固定大小, 然後進行最大值堆疊融合。
    """
    '''
    當目標輸出為 128*128 的時候, 這些位置有最平均的對應像素座標
            x       y
    左眼:   39.4799 40.2799
    右眼:   85.9613 38.7062
    鼻子:   63.6415 63.6473
    左嘴角: 45.6705 89.9648
    右嘴角: 83.9000 88.6898
    ----------------------
    嘴中間: 64.7803 89.3250
    '''
    def __init__(self ):
        super(LocalFuser,self).__init__()
    def forward( self , f_left_eye , f_right_eye , f_nose , f_mouth):
        """
        f: feature
        但傳進來的不一定是「特徵圖」, 也可能是原始圖或 Generator 所產生的 fake 圖

        Args:
            f_left_eye (torch.Tensor): 左眼的特徵圖, 形狀為 (batch_size, channels, height, width)
            f_right_eye (torch.Tensor): 右眼的特徵圖, 形狀為 (batch_size, channels, height, width)
            f_nose (torch.Tensor): 鼻子的特徵圖, 形狀為 (batch_size, channels, height, width)
            f_mouth (torch.Tensor): 嘴巴的特徵圖, 形狀為 (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: 融合後的特徵圖, 形狀為 (batch_size, channels, 128, 128)
        """

        # 指定眼睛、鼻子、嘴部局部特徵的「空間」寬高
        EYE_WIDTH , EYE_HEIGHT = 40 , 40 
        NOSE_WIDTH , NOSE_HEIGHT = 40 , 32
        MOUTH_WITDH , MOUTH_HEIGHT = 48 , 32
        # 指定輸出圖片的大小為 128*128
        IMG_SIZE = 128
        # 填充 4 個部位特徵到指定的位置
        f_left_eye = torch.nn.functional.pad(f_left_eye , (39 - EYE_WIDTH//2  - 1 ,IMG_SIZE - (39 + EYE_WIDTH//2 - 1) ,40 - EYE_HEIGHT//2 - 1, IMG_SIZE - (40 + EYE_HEIGHT//2 - 1)))
        f_right_eye = torch.nn.functional.pad(f_right_eye,(86 - EYE_WIDTH//2  - 1 ,IMG_SIZE - (86 + EYE_WIDTH//2 - 1) ,39 - EYE_HEIGHT//2 - 1, IMG_SIZE - (39 + EYE_HEIGHT//2 - 1)))
        f_nose = torch.nn.functional.pad(f_nose,          (64 - NOSE_WIDTH//2 - 1 ,IMG_SIZE - (64 + NOSE_WIDTH//2 -1) ,64 - NOSE_HEIGHT//2- 1, IMG_SIZE - (64 + NOSE_HEIGHT//2- 1)))
        f_mouth = torch.nn.functional.pad(f_mouth,        (65 - MOUTH_WITDH//2 -1 ,IMG_SIZE - (65 + MOUTH_WITDH//2 -1),89 - MOUTH_HEIGHT//2-1, IMG_SIZE - (89 + MOUTH_HEIGHT//2-1)))
        # 將所有局部特徵圖堆疊在一起，並取最大值
        return torch.max( torch.stack( [ f_left_eye , f_right_eye , f_nose , f_mouth] , dim = 0  ) , dim = 0 )[0]

class GlobalPathway(nn.Module):
    def __init__(self, zdim , local_feature_layer_dim = 64 , use_batchnorm = True , use_residual_block = True , scaling_factor = 1.0 , FM_multiplier = 1.0):
        """
        GlobalPathway 模型, 用於提取和重建整張圖像的全局特徵
        
        Args:
            zdim (int): 隨機噪聲向量的維度
            local_feature_layer_dim (int): 局部特徵層的維度, 默認為 64
            use_batchnorm (bool): 是否使用批次正規化, 默認為 True
            use_residual_block (bool): 是否使用殘差塊, 默認為 True
            scaling_factor (float): 殘差塊的縮放因子, 默認為 1.0
            fm_mult (float): 特徵圖數量的乘數, 默認為 1.0
        """
        super(GlobalPathway,self).__init__()
        # 設定編碼器和解碼器的特徵圖數量
        n_FM_encoder = [64, 64, 128, 256, 512]         # 編碼器的特徵圖數量
        n_FM_decoder = [64, 32, 16, 8]         # 解碼器的特徵圖數量
        n_FM_decoder_enhance_features = [512, 256, 128, 64] # 殘差連接塊的特徵圖數量
        n_FM_decoder_conv = [64, 32]                   # 解碼器上採樣後的特徵處理層的特徵圖數量

        n_FM_encoder = EMaC2I(n_FM_encoder , FM_multiplier)
        n_FM_decoder = EMaC2I( n_FM_decoder , FM_multiplier )
        n_FM_decoder_enhance_features = EMaC2I( n_FM_decoder_enhance_features , FM_multiplier )
        n_FM_decoder_conv = EMaC2I( n_FM_decoder_conv , FM_multiplier )
        
        # 參數賦值
        self.zdim = zdim
        self.use_residual_block = use_residual_block
        ############### 5 個編碼器層, 每個編碼器層都為 1 個 NN 層和 1 個殘差連接塊 ###############
        # 第一層卷積和殘差塊, 保持空間尺寸不變
        # (batch_size, 3, 128, 128) -> (batch_size, n_FM_encoder[0], 128, 128)
        self.conv0 = sequential( conv(               3 , n_FM_encoder[0] , 7 , 1 , 3 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm ),
                                 ResidualBlock(  64 ,  64 , 7 , 1 , 3 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor ) )
        # 第二層卷積和殘差塊, 空間尺寸減半
        # (batch_size, n_FM_encoder[0], 128, 128) -> (batch_size, n_FM_encoder[1], 64, 64)
        self.conv1 = sequential( conv( n_FM_encoder[1] , n_FM_encoder[1] , 5 , 2 , 2 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm ),
                                 ResidualBlock(  64 ,  64 , 5 , 1 , 2 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor ) )
        # 第三層卷積和殘差塊, 空間尺寸再減半
        # (batch_size, n_FM_encoder[1], 64, 64) -> (batch_size, n_FM_encoder[2], 32, 32)
        self.conv2 = sequential( conv( n_FM_encoder[1] , n_FM_encoder[2] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm ),
                                 ResidualBlock( 128 , 128 , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , scaling_factor = scaling_factor ) )
        # 第四層卷積和殘差塊, 空間尺寸再次減半
        # (batch_size, n_FM_encoder[2], 32, 32) -> (batch_size, n_FM_encoder[3], 16, 16)
        self.conv3 = sequential( conv( n_FM_encoder[2] , n_FM_encoder[3] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm ),
                                 ResidualBlock( 256 , 256 , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor ) )
        # 第五層卷積和殘差塊, 空間尺寸再次減半, 這邊堆疊多個殘差連接塊
        # (batch_size, n_FM_encoder[3], 16, 16) -> (batch_size, n_FM_encoder[4], 8, 8)
        self.conv4 = sequential( conv( n_FM_encoder[3] , n_FM_encoder[4] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm ),
                                 *[ ResidualBlock( 512 , 512 , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , is_bottleneck = False , scaling_factor = scaling_factor) for i in range(4) ])
        # 全連接層, 將 8x8 的特徵圖展平為 1 維向量的張量
        # (batch_size, n_FM_encoder[4], 8, 8) -> (batch_size, 512)
        self.fc1 = nn.Linear( n_FM_encoder[4]*8*8 , 512)
        # (batch_size, 1, 512) -> (batch_size, 256)
        self.fc2 = nn.MaxPool1d( 2 , 2 , 0)
        
        ############### 4 個解碼器層, 每個解碼器層都為... ###############
        # (batch_size, 256 + zdim, 1, 1) -> (batch_size, n_FM_decoder[0]:64 , 8, 8)
        self.deconv_8    = deconv( 256 + self.zdim , n_FM_decoder[0] , 8 , 1 , 0 , 0 , "kaiming" , nn.ReLU() , use_batchnorm )
        # (batch_size, n_FM_decoder[0]:64, 8, 8) -> (batch_size, n_FM_decoder[1]:32, 32, 32)
        self.deconv_32   = deconv( n_FM_decoder[0] , n_FM_decoder[1] , 3 , 4 , 0 , 1 , "kaiming" , nn.ReLU() , use_batchnorm )
        # (batch_size, n_FM_decoder[1]:32, 32, 32) -> (batch_size, n_FM_decoder[2]:16, 64, 64)
        self.deconv_64   = deconv( n_FM_decoder[1] , n_FM_decoder[2] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm )
        # (batch_size, n_FM_decoder[2]:16, 64, 64) -> (batch_size, n_FM_decoder[3]:8, 128, 128)
        self.deconv_128  = deconv( n_FM_decoder[2] , n_FM_decoder[3] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm )

        #####
        # 參考 TP-GAN 論文的結構, 需要考慮連結前面的編碼器的輸入, 這邊開始都會先
        # 1. 定義輸入的維度
        # 2. 過殘差連接塊   before_select 系列來進行解碼器與編碼器的特徵圖混合
        # 3. 過多個殘差連接塊 reconstruct 系列來進一步細化和增強特徵

        # 取得 deconv_8 ( 64*8*8 ) 和 conv4 ( 512*8*8 ) 的 channel 數相加w
        dim8 = self.deconv_8.out_channels + self.conv4.out_channels
        # 將 encoder 和 decoder 的特徵圖融合
        self.add_conv_and_deconv_8 = ResidualBlock( dim8 , dim8 , 2 , 1 , padding = [1,0,1,0] , activation = nn.LeakyReLU() )
        # 將融合的特徵圖丟入複數殘差塊中進行進一步的特徵細化
        self.enhance_features_8 = sequential( *[ResidualBlock( dim8 , dim8 , 2 , 1 , padding = [1,0,1,0] , activation = nn.LeakyReLU() ) for i in range(2)] )
        # (batch_size, 576, 8, 8) -> (batch_size, 512, 16, 16)
        self.upsample_16 = deconv( self.enhance_features_8.out_channels , n_FM_decoder_enhance_features[0] , 3 , 2 , 1 , 1, 'kaiming' , nn.ReLU() , use_batchnorm )
        
        # 取得 conv3 ( 256*16*16 ) 的 256
        dim16 = self.conv3.out_channels
        self.add_conv_and_deconv_16 = ResidualBlock( dim16 , activation =nn.LeakyReLU() )
        # (batch_size, 1024, 16, 16)
        self.enhance_features_16 = sequential( *[ResidualBlock( self.upsample_16.out_channels + self.add_conv_and_deconv_16.out_channels , activation = nn.LeakyReLU() )for i in range(2) ] )
        # (batch_size, 1024, 16, 16) -> (batch_size, 256, 16, 16)
        self.upsample_32 = deconv( self.enhance_features_16.out_channels , n_FM_decoder_enhance_features[1] , 3 , 2 , 1 , 1, 'kaiming' , nn.ReLU() , use_batchnorm )
        
        # 取得 conv2 ( 128*32*32 ) + deconv_32 ( 32*32*32 ) 的 128 + 32 = 160
        dim32 = self.conv2.out_channels + self.deconv_32.out_channels
        self.add_conv_and_deconv_32 = ResidualBlock( dim32 , activation = nn.LeakyReLU() )
        # (batch_size, 416, 32, 32)
        self.enhance_features_32 = sequential( *[ResidualBlock( self.upsample_32.out_channels + self.add_conv_and_deconv_32.out_channels , activation = nn.LeakyReLU()) for i in range(2) ] )
        #self.decoded_img32 = conv( self.enhance_features_32.out_channels , 3 , 3 , 1 , 1 , None ,  None )
        # (batch_size, 416, 32, 32) -> (batch_size, 128, 64, 64)
        self.upsample_64 = deconv( self.enhance_features_32.out_channels , n_FM_decoder_enhance_features[2] , 3 , 2 , 1 , 1 , 'kaiming' , nn.ReLU() , use_batchnorm )
        
        # 取得 conv1 ( 64*64*64 ) + deconv_64 ( 16*64*64 ) 的 64 + 16 = 80
        dim64 = self.conv1.out_channels + self.deconv_64.out_channels
        self.add_conv_and_deconv_64 = ResidualBlock(  dim64 , kernel_size =  5 , activation = nn.LeakyReLU()   ) 
        # (batch_size, 208, 64, 64)
        self.enhance_features_64 = sequential( *[ResidualBlock( self.upsample_64.out_channels + self.add_conv_and_deconv_64.out_channels , activation = nn.LeakyReLU()) for i in range(2) ] )
        #self.decoded_img64 = conv( self.enhance_features_64.out_channels , 3 , 3 , 1 , 1 , None ,  None )
        # (batch_size, 208, 64, 64) -> (batch_size, 64, 128, 128)
        self.upsample_128 = deconv( self.enhance_features_64.out_channels , n_FM_decoder_enhance_features[3] , 3 , 2 , 1 , 1 , 'kaiming' , nn.ReLU() , use_batchnorm )
        
        # 取得 conv0 ( 64*128*128 ) + deconv_128 ( 64*128*128 ) 的 64 + 64 = 128
        dim128 = self.conv0.out_channels + self.deconv_128.out_channels
        self.add_conv_and_deconv_128 = ResidualBlock( dim128  , kernel_size = 7 , activation = nn.LeakyReLU()  )
        # (batch_size, 256 + 3, 128, 128)
        self.enhance_features_128 = sequential( *[ResidualBlock( self.upsample_128.out_channels + self.add_conv_and_deconv_128.out_channels + local_feature_layer_dim + 3 , kernel_size = 5 , activation = nn.LeakyReLU())] )
        
        # 對特徵維度進行捲積處理, 並轉換特徵維度, (batch_size, 256 + 3, 128, 128) -> (batch_size, 64, 128, 128)
        self.conv5 = sequential( conv( self.enhance_features_128.out_channels , n_FM_decoder_conv[0] , 5 , 1 , 2 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) ,
                                 ResidualBlock(n_FM_decoder_conv[0] , kernel_size = 3 , activation = nn.LeakyReLU() ))
        # (batch_size, 64, 128, 128) -> (batch_size, 128, 128, 128)
        self.conv6 = conv( n_FM_decoder_conv[0] , n_FM_decoder_conv[1] , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm )
        # 進行圖片輸出, (batch_size, 128, 128, 128) -> (batch_size, 3, 128, 128)
        self.decoded_img128 = conv( n_FM_decoder_conv[1] , 3 , 3 , 1 , 1 , None , activation = None )

    def forward(self, I128, local_fake_image, local_feature, z ):
        ##### 編碼過程 #####
        conv0 = self.conv0( I128)#128x128
        conv1 = self.conv1( conv0)#64x64
        conv2 = self.conv2( conv1)#32x32
        conv3 = self.conv3( conv2)#16x16
        conv4 = self.conv4( conv3)#8x8

        fc1 = self.fc1( conv4.view( conv4.size()[0] , -1 ))
        fc2 = self.fc2( fc1.view( fc1.size()[0] , -1 , 2  )).view( fc1.size()[0] , -1 ) 

        ##### 解碼過程 #####
        deconv_8   = self.deconv_8( torch.cat([fc2,z] , 1).view( fc2.size()[0] , -1 , 1 , 1 )  )
        deconv_32  = self.deconv_32( deconv_8)
        deconv_64  = self.deconv_64( deconv_32)
        deconv_128 = self.deconv_128( deconv_64)
        # 將 encoder 和 decoder 的特徵圖相加
        add_conv_and_deconv_8 = self.add_conv_and_deconv_8( torch.cat( [deconv_8, conv4] , 1 ) )
        # 將相加的特徵圖進行多殘差快的進一步細化處理
        enhance_features_8 = self.enhance_features_8( add_conv_and_deconv_8 )
        assert enhance_features_8.shape[2] == 8
        # 將處理後的特徵圖進行空間尺寸的提升
        upsample_16 = self.upsample_16( enhance_features_8 )


        add_conv_and_deconv_16 = self.add_conv_and_deconv_16( conv3 )
        enhance_features_16 = self.enhance_features_16( torch.cat( [upsample_16 , add_conv_and_deconv_16] , 1 ) )
        assert enhance_features_16.shape[2] == 16
        upsample_32 = self.upsample_32( enhance_features_16 )
        
        add_conv_and_deconv_32 = self.add_conv_and_deconv_32( torch.cat( [deconv_32 , conv2] ,  1 ) )
        enhance_features_32 = self.enhance_features_32( torch.cat( [upsample_32 , add_conv_and_deconv_32] , 1 ) )
        #decoded_img32 = self.decoded_img32( enhance_features_32 )
        #assert decoded_img32.shape[2] == 32
        upsample_64 = self.upsample_64( enhance_features_32 )
        
        add_conv_and_deconv_64 = self.add_conv_and_deconv_64( torch.cat( [deconv_64 , conv1] , 1 ) )
        enhance_features_64 = self.enhance_features_64( torch.cat( [upsample_64 , add_conv_and_deconv_64] , 1 ) )
        #decoded_img64 = self.decoded_img64( enhance_features_64 )
        #assert decoded_img64.shape[2] == 64
        upsample_128 = self.upsample_128( enhance_features_64 )
        
        add_conv_and_deconv_128 = self.add_conv_and_deconv_128( torch.cat( [deconv_128 , conv0 , I128 ] , 1 ) )
        enhance_features_128 = self.enhance_features_128( torch.cat( [upsample_128 , add_conv_and_deconv_128 , local_feature , local_fake_image ] , 1 ) )
        
        conv5 = self.conv5( enhance_features_128 )
        conv6 = self.conv6( conv5 )
        decoded_img128 = self.decoded_img128( conv6 )
        return decoded_img128 , fc2
        
class FeaturePredict(nn.Module):
    """
    FeaturePredict 模型, 用於從全局特徵中進行分類預測。
    
    Args:
        num_classes (int): 類別數目, 用於分類的輸出維度
        global_feature_layer_dim (int): 全局特徵的維度, 默認為 256
        dropout (float): Dropout 機率, 用於防止過擬合, 默認為 0.3
    """
    def __init__(self ,  num_classes , global_feature_layer_dim = 256 , dropout = 0.3):
        super(FeaturePredict,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(global_feature_layer_dim , num_classes )
    def forward(self ,x ,use_dropout):
        if use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x
        
class Generator(nn.Module):
    def __init__(self, zdim , num_classes , use_batchnorm = True , use_residual_block = True):
        """
        Generator 模型, 包含局部和全局特徵模型, 用於生成以假亂真的 fake 的人臉圖像
        
        Args:
            zdim (int): 隨機噪聲向量的維度
            num_classes (int): 用於特徵預測的類別數
            use_batchnorm (bool): 是否使用批次正規化, 默認為 True
            use_residual_block (bool): 是否使用殘差塊, 默認為 True
        """
        super(Generator,self).__init__()
        # 定義局部特徵模型, 用於提取局部特徵
        self.local_pathway_left_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_right_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_nose  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_mouth  = LocalPathway(use_batchnorm = use_batchnorm)
        # 定義全局特徵模型, 用於生成全局特徵圖像
        self.global_pathway = GlobalPathway(zdim , use_batchnorm = use_batchnorm , use_residual_block = use_residual_block)
        # 定義局部特徵融合模型
        self.local_fuser    = LocalFuser()
        # 定義特徵預測模塊, 並傳入需要預測的特徵數量
        self.feature_predict = FeaturePredict(num_classes)

    def forward( self, I128 , left_eye , right_eye , nose , mouth , z , use_dropout ):
        """
        Args:
            I128 (torch.Tensor): 輸入的128x128圖像。
            left_eye (torch.Tensor): 左眼的局部圖像。
            right_eye (torch.Tensor): 右眼的局部圖像。
            nose (torch.Tensor): 鼻子的局部圖像。
            mouth (torch.Tensor): 嘴巴的局部圖像。
            z (torch.Tensor): 隨機噪聲向量。
            use_dropout (bool): 是否使用dropout。
        
        Returns:
            Tuple: 包含生成的 128x128 fake 圖像、特徵預測、局部視覺圖、左眼生成圖、右眼生成圖、鼻子生成圖、嘴巴生成圖、局部輸入圖。
        """
        ##### 局部圖像過程 #####
        # 提取局部圖片與局部特徵
        left_eye_fake_image  , left_eye_fake_feature  = self.local_pathway_left_eye( left_eye)
        right_eye_fake_image , right_eye_fake_feature = self.local_pathway_right_eye( right_eye)
        nose_fake_image      , nose_fake_feature      = self.local_pathway_nose( nose)
        mouth_fake_image     , mouth_fake_feature     = self.local_pathway_mouth( mouth)

        # 融合局部圖片與局部特徵
        fused_local_feature       = self.local_fuser( left_eye_fake_feature , right_eye_fake_feature , nose_fake_feature , mouth_fake_feature )
        fused_local_fake_image    = self.local_fuser( left_eye_fake_image , right_eye_fake_image , nose_fake_image , mouth_fake_image )
        fused_local_origin_4_part = self.local_fuser( left_eye , right_eye , nose , mouth ) # 將真實局部圖片也進行融合


        ##### 全局圖像過程 #####
        # 通過全局特徵模型生成以假亂真圖片和編碼後的 8*8 特徵圖
        I128_fake , encoder_feature = self.global_pathway( I128 , fused_local_fake_image , fused_local_feature , z)
        # 使用特徵圖去預測分類
        encoder_predict = self.feature_predict( encoder_feature , use_dropout )
        
        return I128_fake , encoder_predict , fused_local_fake_image , left_eye_fake_image , right_eye_fake_image , nose_fake_image , mouth_fake_image , fused_local_origin_4_part

class Discriminator(nn.Module):
    def __init__(self, use_batchnorm = False , FM_multiplier = 1.0):
        """
        Discriminator 模型, 用於判別輸入的圖像是真實圖像還是生成圖像
        
        Args:
            use_batchnorm (bool): 是否使用批次正規化, 默認為 False
            FM_multiplier (float): 特徵圖數量的乘數, 用於調整每層的特徵圖數量, 默認為 1.0
        """
        super(Discriminator,self).__init__()
        layers = []
        # 定義每層的特徵圖數量, 第一個 3 代表輸入的 RGB 圖像, 其餘的分別代表每層的特徵圖數量。
        n_Fmap = [3, 64, 128, 256, 512, 512] 
        n_Fmap = EMaC2I( n_Fmap , FM_multiplier )
        for i in range( len( n_Fmap ) - 1 ):
            #layers.append( conv( n_fmap[i] , n_fmap[i+1] , kernel_size = 4 , stride = 2 , padding = 1 , init = "kaiming" , activation = nn.LeakyReLU(1e-2) ) )
            # 添加卷積層
            layers.append( conv( n_Fmap[i] , n_Fmap[i+1] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm  ) )
            # 從第四層開始添加殘差連接塊
            if i >=3:
                layers.append( ResidualBlock( n_Fmap[i+1] , activation = nn.LeakyReLU() ) )
        # 添加最後一層的卷積層, 並將輸出通道數變為 1, 形狀為 (batch_size, 1, H, W), 此並非針對單一張圖像的真實度判別, 而是產生 H*W 的局部真實度數值
        layers.append( conv( n_Fmap[-1] , 1 , kernel_size = 3 ,  stride = 1 , padding = 1 , init = None ,activation =  None ))
        self.model = sequential( *layers )

    def forward(self, x):
        return self.model(x)