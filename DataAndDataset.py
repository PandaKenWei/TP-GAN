import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from math import floor
from UtilityMethods import get_5_landmarks_pixal_position
import os

def process(img  , landmarks_5pts):
    """
    Args:
        img(PIL.Image.Image): 要處理的圖片，預期為一個 PIL 影像對象
        landmarks_5pts(numpy.ndarray):一個形狀為 (5, 2) 的陣列，包含五個主要的臉部標誌點
            - 這 5 個點分別代表兩個眼睛的中心點、鼻尖以及兩個嘴角的一組 (x, y)
                - array([[ x1,  y1],  # 左眼中心座標
                         [ x2,  y2],  # 右眼中心座標
                         [ x3,  y3],  # 鼻尖座標
                         [ x4,  y4],  # 左嘴角座標
                         [ x5,  y5]]) # 右嘴角座標
            - 其中，兩個嘴角的位置會在處理過程中取平均，來代表整個嘴部中心位置
    Return:
        batch(dict): 用於儲存剪裁後的 4 個臉部區域 ( 左眼、右眼、鼻子、嘴部 )
            - { "left_eye": PIL.Image.Image,
                "right_eye": PIL.Image.Image,
                "nose": PIL.Image.Image,
                "mouth":PIL.Image.Image
                }
    """

    batch = {}
    # 定義剪裁區域的 4 個名稱
    name = ['left_eye','right_eye','nose','mouth']
    # 指定 4 個剪裁區域的剪裁尺寸
    patch_size = {
            'left_eye':(40,40),
            'right_eye':(40,40),
            'nose':(40,32),
            'mouth':(48,32),
    }
    # 取得嘴部中心座標 (x, y)
    landmarks_5pts[3,0] =  (landmarks_5pts[3,0] + landmarks_5pts[4,0]) / 2.0
    landmarks_5pts[3,1] = (landmarks_5pts[3,1] + landmarks_5pts[4,1]) / 2.0

    # 依照定義的變數 name 依序對特定區域使用 img.crop 函數進行剪裁
    for i in range(4):
        # 使用 floor 確保剪裁的座標為整數
        x = floor(landmarks_5pts[i,0])
        y = floor(landmarks_5pts[i,1])
        # crop 函數接受 1 個 4 空間的元組，為 (left, upper, right, lower)
        batch[ name[i] ] = img.crop( (x - patch_size[ name[i] ][0]//2 + 1 ,
                                      y - patch_size[ name[i] ][1]//2 + 1 ,
                                      x + patch_size[ name[i] ][0]//2 + 1 ,
                                      y + patch_size[ name[i] ][1]//2 + 1 ) )

    return batch


##### Pretrain Dataset 和其輔助函式 #####
class PretrainDataset( Dataset):
    def __init__( self , txt_name, data_root_dir):
        """
        Args:
            txt_name (string): 包含圖像名稱和標籤的 txt 文件路徑
            data_root_dir (string): 包含圖像、標籤檔案的根目錄路徑
        """
        self.labels = _getPretrainLabelGroups(txt_name, data_root_dir)
        self.image_names = _getPretrainImageFullPaths(data_root_dir)
        
    def __len__( self ):
        return len( self.image_names )
    def __getitem__( self, idx ):
        # 獲取圖片的完整路徑
        img_path = self.image_names[idx]

        # 從路徑名稱中提取圖片名稱
        img_name = img_path.split('\\')[-1]

        # 讀取圖片
        image = Image.open( img_path ).convert('RGB')  # 確保圖片是RGB格式

        # 獲取對應圖片名的的標籤
        label_groups = self.labels[img_name]

        # 將 PIL Image 轉換為 Tensor
        transform = transforms.ToTensor()
        image = transform( image )

        # 將每個座標轉換成獨立的張量，並展平為一維張量
        label_tensors = torch.tensor([
            label_groups[0][0], label_groups[0][1],  # 左眼
            label_groups[1][0], label_groups[1][1],  # 右眼
            label_groups[2][0], label_groups[2][1],  # 鼻子
            label_groups[3][0], label_groups[3][1]   # 嘴巴
        ], dtype=torch.float32)

        return image, label_tensors

def _getPretrainLabelGroups(txt_name, data_root_dir):
    """
    解析 label 檔案並返回一個字典, 每個鍵是圖片名、值是臉部特徵座標的 list, 包括
        - 左眼
        - 右眼
        - 鼻子
        - 嘴巴
        中心點座標

    Args:
        txt_name (string): 包含圖像名稱和標籤的 label 檔案 txt 文件路徑
        data_root_dir (string): Pretrain 資料集的根目錄路徑
    
    Returns:
        dict: {
            "image name": [(left_eye_x, left_eye_y),
                           (right_eye_x, right_eye_y),
                           (nose_x, nose_y),
                           (mouth_x, mouth_y)]
        }
    """
    # 組合 label 檔案的完整路徑
    label_file_path = os.path.join(data_root_dir, txt_name)
    # 定義要返回的 label 字典
    labels_dict = {}

    with open( label_file_path , 'r' ) as file:
        next(file)  # 跳過第一行 ( 圖片數量 )
        next(file)  # 跳過第二行  (列名標頭 )
        # 逐筆處理標籤成指定格式
        for line in file:
            # 使用「空白」做為分隔條件, 切分出 11 組數據
            parts = line.split()
            # 取得該筆標籤對應的圖片名稱
            image_name = parts[0]
            # 取得眼部中心像素 x 與 y 座標
            lefteye_x, lefteye_y = int( parts[1] ), int( parts[2] )
            righteye_x, righteye_y = int( parts[3] ), int( parts[4] )
            # 取得鼻子中心像素 x 與 y 座標
            nose_x, nose_y = int( parts[5] ), int( parts[6] )
            # 取得嘴部像素 x 與 y 座標
            leftmouth_x, leftmouth_y = int( parts[7] ), int( parts[8] )
            rightmouth_x, rightmouth_y = int( parts[9] ), int( parts[10] )
            
            # 計算嘴巴中點座標
            mouth_x = ( leftmouth_x + rightmouth_x ) // 2
            mouth_y = ( leftmouth_y + rightmouth_y ) // 2
            
            # 存儲到字典
            labels_dict[ image_name ] = [
                ( lefteye_x, lefteye_y ),
                ( righteye_x, righteye_y ),
                ( nose_x, nose_y ),
                ( mouth_x, mouth_y )
            ]

    return labels_dict

def _getPretrainImageFullPaths(data_root_path):
    """
    遍歷資料集根目錄下的所有 jpg 檔案, 並返回一個 list, 包含依照順序儲存的完整路徑圖片名稱

    Args:
        data_root_path (string): Pretrain 資料集的根目錄路徑

    Returns:
        list: 包含所有 jpg 文件的完整路徑
    """

    # 定義要返回的 list
    image_paths = []

    # 遍歷根目錄及其子目錄
    for root, _, files in os.walk(data_root_path):
        for file in files:
            if file.lower().endswith('.jpg'):  # 確保只處理 jpg 文件
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    return image_paths

# 下面待改
class TrainDataset( Dataset):
    def __init__( self , img_list ):
        """
        Args:
            img_list (list of str): 圖片檔案的路徑 list, 每個檔案名稱包含有關圖像及其前視圖和不同尺寸版本的訊息。
        """
        super(type(self),self).__init__()
        self.img_list = img_list
    def __len__( self ):
        """
        返回數據集的長度，即圖片 list 的總數量
        """
        return len(self.img_list)
    def __getitem__( self , idx ):
        """
        根據索引 idx 獲取並返回對應的圖像數據和標籤

        Return:

        """
        
        batch = {}
        ### 檔案名稱處理
        img_name = self.img_list[idx].split('/')
        img_frontal_name = self.img_list[idx].split('_')
        img_frontal_name[-2] = '051'
        img_frontal_name = '_'.join( img_frontal_name ).split('/')
        batch['img'] = Image.open( '/'.join( img_name ) )
        batch['img32'] = Image.open( '/'.join( img_name[:-2] + ['32x32' , img_name[-1] ] ) )
        batch['img64'] = Image.open( '/'.join( img_name[:-2] + ['64x64' , img_name[-1] ] ) )
        batch['img_frontal'] = Image.open( '/'.join(img_frontal_name) )
        batch['img32_frontal'] = Image.open( '/'.join( img_frontal_name[:-2] + ['32x32' , img_frontal_name[-1] ] ) )
        batch['img64_frontal'] = Image.open( '/'.join( img_frontal_name[:-2] + ['64x64' , img_frontal_name[-1] ] ) )
        patch_name_list = ['left_eye','right_eye','nose','mouth']
        for patch_name in patch_name_list:
            batch[patch_name] = Image.open( '/'.join(img_name[:-2] + ['patch' , patch_name , img_name[-1] ]) ) 
            batch[patch_name+'_frontal'] = Image.open( '/'.join(img_frontal_name[:-2] + ['patch' , patch_name , img_frontal_name[-1] ]) )
        totensor = transforms.ToTensor()

        for k in batch:
            batch[k] = totensor( batch[k] ) 
            batch[k] = batch[k] *2.0 -1.0
            #if batch[k].max() <= 0.9:
            #    print( "{} {} {}".format( batch[k].max(), self.img_list[idx] , k  ))
            #if batch[k].min() >= -0.9:
            #    print( "{} {} {}".format( batch[k].min() , self.img_list[idx] , k ) )

        batch['label'] = int( self.img_list[idx].split('/')[-1].split('_')[0] )
        return batch


class TestDataset( Dataset):
    def __init__( self , img_list , lm_list):
        super(type(self),self).__init__()
        self.img_list = img_list
        self.lm_list = lm_list
        assert len(img_list) == len(lm_list)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self,idx):
        img_name = self.img_list[idx]
        img = Image.open( img_name )

        lm = np.array( self.lm_list[idx].split(' ') , np.float32 ).reshape(-1,2)
        lm = get_5_landmarks_pixal_position( lm )
        for i in range(5):
            lm[i][0] *= 128/img.width 
            lm[i][1] *= 128/img.height
        img = img.resize( (128,128) , Image.LANCZOS)
        batch = process( img , lm )
        batch['img'] = img
        batch['img64'] = img.resize( (64,64) , Image.LANCZOS )
        batch['img32'] = batch['img64'].resize( (32,32) , Image.LANCZOS )
        to_tensor = transforms.ToTensor() 
        for k in batch:
            batch[k] = to_tensor( batch[k] )
            batch[k] = batch[k] * 2.0 - 1.0
        return batch