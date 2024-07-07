"""
定義一些方法，供其他檔案使用
"""

import os
import torch
from torch import optim
import numpy as np
import torchvision.transforms as transforms
import warnings
import PIL.Image as Image
from config import optimizer_param

def getOptimizer( model_parameters , optimizer_name="SGD" ) -> optim.Optimizer:
    """
    根據傳入的 optimizer 字串來回傳相應的 optimizer 物件

    Args:
        model_parameters: 模型的參數, 注意 ! 這邊傳進來的是生成器而不是 list, 因此要先處理
        optimizer_name (string): 傳入的 optimizer 字串, 默認為 SGD
    
    Returns:
        torch.optim.Optimizer: 根據指定的名稱回傳相應的 PyTorch Optimizer 實例, 如果指定的名稱不存在於字典中則默認回傳 SGD Optimizer
    """

    # 將生成器轉換為列表
    model_parameters = list(model_parameters)

    # 定義常用的 Optimizer 字典, 並且使用 config 參數來完成其中物件定義
    optimizers = {
        'SGD': lambda: optim.SGD(model_parameters, lr=optimizer_param['learning_rate'], weight_decay=optimizer_param['weight_decay'], momentum=optimizer_param['momentum'], nesterov=optimizer_param.get('nesterov', False)),
        'Adam': lambda: optim.Adam(model_parameters, lr=optimizer_param['learning_rate'], weight_decay=optimizer_param['weight_decay']),
        'RMSprop': lambda: optim.RMSprop(model_parameters, lr=optimizer_param['learning_rate'], weight_decay=optimizer_param['weight_decay'], momentum=optimizer_param['momentum']),
        'Adagrad': lambda: optim.Adagrad(model_parameters, lr=optimizer_param['learning_rate'], weight_decay=optimizer_param['weight_decay']),
        'Adadelta': lambda: optim.Adadelta(model_parameters, lr=optimizer_param['learning_rate'], weight_decay=optimizer_param['weight_decay'])
    }

    # 根據傳入的 optimizer 名稱查找字典來返回 optimizer 物件
    optimizer = optimizers.get(optimizer_name, optimizers['SGD'])  # 預設為 SGD 如果查找失敗

    return optimizer

def set_requires_grad(parameters, isGrad):
    """
    設定模型參數的 requires_grad 屬性, 這個屬性決定了這些參數在反向傳播時是否需要計算梯度

    Args:
        parameters (iterator): 來自模型的參數迭代器, 通常獲得方式為 model.parameters()
        isGrad (bool): 如果設為 True 則開啟梯度計算；反之如果為 False 則關閉梯度計算

    Example:
        model = SomeModel()
        set_requires_grad(model.parameters(), True)  # 開啟所有模型參數的梯度計算
    """
    for param in parameters:
        param.requires_grad = isGrad

def save_model(model, dir, epoch):
    """
    保存模型到指定目錄

    Args:
        model (torch.nn.Module): 要保存的模型
        dir (str): 保存模型的目錄
        epoch (int): 當前訓練的 epoch
    """

    # 構建模型檔案名
    model_filename = os.path.join( dir, f'model_epoch_{epoch}.pth' )
    
    # 創建目錄如果不存在
    os.makedirs( os.path.dirname( model_filename ), exist_ok=True )
    
    # 保存模型狀態字典
    torch.save( model.state_dict() , model_filename )
    print( f'儲存模型 {model_filename}' )

def save_optimizer(optimizer, model, dir, epoch):
    """
    保存優化器狀態到指定目錄

    Args:
        optimizer (torch.optim.Optimizer): 要保存的優化器
        model (torch.nn.Module): 要保存的模型
        dir (str): 保存優化器狀態的目錄
        epoch (int): 當前訓練的 epoch
    """
    # 構建優化器文件名
    optimizer_filename = os.path.join( dir, f'optimizer_epoch_{epoch}.pth' )
    
    # 創建目錄如果不存在
    os.makedirs( os.path.dirname( optimizer_filename ), exist_ok=True )
    
    # 構建保存的字典
    save_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    
    # 保存優化器狀態字典
    torch.save( save_dict , optimizer_filename )
    print( f'Optimizer saved to {optimizer_filename}' )





def elementwise_multiply_and_cast_to_int(list_x , scalar):
    """
    逐元素地將 list 中的每個數值乘以一個單一數值, 然後將乘積轉換為整數
    用於調整配置參數或應用於需要整數值的場合

    Args:
        list_x (list of float or int): 要被操作的數值 list
        scalar (float or int): 與 list 中每個元素相乘的單一數值

    Returns:
        list of int: 返回一個新的 list, 其中內容為 -> 原始 list 的每個元素都乘以單一數值並轉換為整數
    """
    return [ int( v*scalar ) for v in list_x ]

def resize_tensor(x, size, interpolation =  Image.BILINEAR):
    """
    調整輸入的圖像的張量維度
        - 首先將張量轉換為 PIL 圖像數據, 然後使用指定的插值方法調整大小, 最後將其轉換回張量
        - 處理圖片數據以適配神經網絡模型輸入尺寸要求的方法

    Args:
        x (Tensor): 要調整大小的圖像的張量。
        size (tuple of int): 一個包含新高度和寬度的元組或單個整數, 如果是單個整數, 則高度和寬度會設置為相同的值
        
        interpolation (int): 插值方法, 預設為 PIL.Image.BILINEAR

    Returns:
        Tensor: 調整大小後的圖片張量
    """

    # transforms.Compose 接受一個 list, list 中的每個元素都是一個對圖像的轉換操作
    transform = transforms.Compose([
        transforms.ToPILImage(), # 將輸入的張量轉換為 PIL 圖像數據, 以便後續的大小調整操作
        transforms.Resize(size, interpolation = interpolation ), # 調整 PIL 圖像的大小以符合指定的尺寸, PIL.Image.BILINEAR 為「雙線性插值」
        transforms.ToTensor(), # 將調整大小後的 PIL 圖像轉換回 PyTorch 張量，以便進行進一步的神經網絡模型處理
        ])
    return transform(x)

# 定義 Dlib 的左眼、右眼、鼻子、左嘴角、右嘴角的點位「區間」
five_pts_idx = [ [36,41] , [42,47] , [27,35] , [48,48] , [68,68] ]
def get_5_landmarks_pixal_position(x):
    """
    將一個包含 68 個人臉特徵點的 list 轉換為一個包含 5 個關鍵人臉特徵點的陣列
    每個關鍵特徵點是透過計算指定區間內點的平均位置 (x, y) 獲得的。

    Args:
        x (np.array): 一個形狀為 (68, 2) 的 NumPy list, 包含 68 個人臉特徵點的 x 和 y 坐標

    Returns:
        np.array: 一個形狀為 (5, 2) 的 NumPy list, 包含計算後的 5 個關鍵人臉特徵點的 x 和 y 坐標, 數據類型為 np.float32
    """
    y = []
    for j in range(5):
        y.append( np.mean( x[ five_pts_idx[j][0]:five_pts_idx[j][1] + 1] , axis = 0  ) )
        
    return np.array( y , np.float32)

