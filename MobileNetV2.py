#borrowed from https://github.com/tonylins/pytorch-mobilenet-v2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import general, pretrain

class SSDHead(nn.Module):
    def __init__(self, num_of_out_classes=4):
        super( SSDHead, self ).__init__()

        # 預測目標分類種類
        self.num_of_out_classes = num_of_out_classes
        # 預測目標的輸出結果 -> 2 種為中心的 x、y
        self.num_of_out_location = 2

        # 定義座標 ( 原為物體框定位 ) 的輸出層和分類輸出層
        self.location_layer = nn.ModuleList()
        self.classification_layer = nn.ModuleList()

        ##### 這個區塊復刻原始論文的 6 層特徵圖 #####
        """
        特徵圖中的 6 層, 錨框數量分別為 4, 6, 6, 6, 6, 6
        因此輸出通道數為 錨框數量 * 輸出類別
        """
        self.location_layer       += [ nn.Conv2d(    96, 4 * self.num_of_out_location , kernel_size=3 , padding=1 ) ]
        self.classification_layer += [ nn.Conv2d(    96, 4 * self.num_of_out_classes  , kernel_size=3 , padding=1 ) ]
        
        self.location_layer       += [ nn.Conv2d( 1280 , 6 * self.num_of_out_location , kernel_size=3 , padding=1 ) ]
        self.classification_layer += [ nn.Conv2d( 1280 , 6 * self.num_of_out_classes  , kernel_size=3 , padding=1 ) ]
        
        self.location_layer       += [ nn.Conv2d(  512 , 6 * self.num_of_out_location , kernel_size=3 , padding=1 ) ]
        self.classification_layer += [ nn.Conv2d(  512 , 6 * self.num_of_out_classes  , kernel_size=3 , padding=1 ) ]
        
        self.location_layer       += [ nn.Conv2d(  256 , 6 * self.num_of_out_location , kernel_size=3 , padding=1 ) ]
        self.classification_layer += [ nn.Conv2d(  256 , 6 * self.num_of_out_classes  , kernel_size=3 , padding=1 ) ]
        
        self.location_layer       += [ nn.Conv2d(  256 , 6 * self.num_of_out_location , kernel_size=3 , padding=1 ) ]
        self.classification_layer += [ nn.Conv2d(  256 , 6 * self.num_of_out_classes  , kernel_size=3 , padding=1 ) ]
        
        self.location_layer       += [ nn.Conv2d(  128 , 6 * self.num_of_out_location , kernel_size=3 , padding=1 ) ]
        self.classification_layer += [ nn.Conv2d(  128 , 6 * self.num_of_out_classes  , kernel_size=3 , padding=1 ) ]

    def forward( self, features ):
        """
        計算每個特徵圖的定位和分類預測

        Args:
            feature (list of torch.Tensor): 來自不同特徵圖層的 feature list

        Returns:
            locations (torch.Tensor): 合併所有特徵圖的定位預測結果, 形狀為 (batch_size, num_total_predictions * 2)。
            classifications (torch.Tensor): 合併所有特徵圖的分類預測結果，形狀為 (batch_size, num_total_predictions * num_of_out_classes)。
        """

        locations = []
        classifications = []

        # 遍歷所有特徵圖層, 依照 idx 分別計算「定位」和「分類」的預測結果
        for idx, x in enumerate( features ):
            location = self.location_layer[ idx ]( x ).permute( 0, 2, 3, 1 ).contiguous()
            # 得到 ( 1, n , 2 ), 2 代表一組 x, y
            location = location.view( location.size( 0 ) , -1 , self.num_of_out_location )
            # 使用 ReLU 確保座標值為非負
            location = torch.relu( location )
            locations.append( location )

            classification = self.classification_layer[ idx ]( x ).permute( 0, 2, 3, 1 ).contiguous()
            # 得到 ( 1, n , 5 ), 5 代表 5 種分類標籤
            classification = classification.view( classification.size( 0 ) , -1 , self.num_of_out_classes )
            classifications.append( classification )

        # 將所有特徵圖的預測結果展平並拼接在一起
        locations = torch.cat( locations , 1 )
        classifications = torch.cat( classifications , 1 )

        return locations, classifications

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
        # 引入 SSDHead 改裝模組
        self.ssd_head = SSDHead( 4+1 ) # 左眼、右眼、鼻子、嘴吧  + 背景 ( 其他 )

        # 額外定義「特徵提取層」, 以適應改裝的 SSDHead
        self.extra_layers = nn.ModuleList([
            nn.Conv2d(1280, 512, kernel_size=1),                     # 降通道數
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 降空間尺寸 ( 第三層 )
            nn.Conv2d(512, 256, kernel_size=1),                      # 降通道數
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 降空間尺寸 ( 第四層 )
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 降空間尺寸 ( 第五層 )
            nn.Conv2d(256, 128, kernel_size=1),                      # 降通道數
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 降空間尺寸 ( 第六層 )
        ])

        self._initialize_weights()

    def forward(self, x , use_dropout=False):
        
        # 用於儲存 6 個不同尺度的臨時特徵圖, 用於不同解析度的物件判斷
        features = []

        # 第一層
        x = self.conv1(x)
        # 7 個 Bottleneck 共計 17 層 Inverted Residual Block
        for idx, bottleneck in enumerate( self.bottlenecks ):
            x = bottleneck(x)
            # idx = 12, 第五個 bottleneck 的結果為原始論文 SSD 的第一個 feature
            if idx == 12:
                features.append(x)

        # 最後一層
        x = self.conv2(x)
        # 這邊為第二個 feature
        features.append(x)

        # 將特徵圖進行進一步的提取
        for idx, extra_layer in enumerate( self.extra_layers ):
            x = extra_layer(x)
            # 將特定層加入特徵圖中
            if idx in [1, 3, 4, 6]:
                features.append(x)
        
        # 將 6 個不同的尺度的特徵圖過 SSDHead, 取得定位框與類別預測的資訊
        locations, classifications = self.ssd_head( features )
        
        return locations, classifications

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

    def non_maximum_suppression(self, points, scores, distance_threshold):
        """
        基於預測點的非最大抑制

        Args:
            points (torch.Tensor): 預測點張量，形狀為 (N, 2)，每行表示 [x, y]
            scores (torch.Tensor): 置信度分數張量，形狀為 (N,)
            distance_threshold (float): 距離閾值，用於判定是否移除重疊點

        Returns:
            keep (list): 保留的預測點索引
        """
        if points.numel() == 0:
            return []

        # 根據分數進行排序
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            current_point = points[i]
            remaining_points = points[order[1:]]

            # 計算歐幾里得距離
            distances = torch.norm(remaining_points - current_point, dim=1)

            # 保留距離大於閾值的點
            indices = (distances > distance_threshold).nonzero(as_tuple=False).squeeze()
            order = order[indices + 1]

        return keep

    def find_best_coordinates(self, locations, classifications, distance_threshold=15.0):
        """
        找出每個部位的最佳預測坐標

        Args:
            locations (torch.Tensor): 預測位置張量，形狀為 (batch_size, num_total_predictions, 10)
            classifications (torch.Tensor): 預測分類張量，形狀為 (batch_size, num_total_predictions, 5)
            distance_threshold (float): 距離閾值，用於非最大抑制

        Returns:
            dict: 包含每個部位最佳預測坐標的字典
        """
        # 分離每個部位的坐標
        lefteye_coordinates = locations[:, :, 0:2]
        righteye_coordinates = locations[:, :, 2:4]
        nose_coordinates = locations[:, :, 4:6]
        leftmouth_coordinates = locations[:, :, 6:8]
        rightmouth_coordinates = locations[:, :, 8:10]

        # 分離每個部位的分數
        lefteye_scores = classifications[:, :, 0]
        righteye_scores = classifications[:, :, 1]
        nose_scores = classifications[:, :, 2]
        leftmouth_scores = classifications[:, :, 3]
        rightmouth_scores = classifications[:, :, 4]

        # 排除背景類別（假設背景類別是最後一個）
        background_index = 4

        # 找出非背景的最佳索引
        best_lefteye_index = self.non_maximum_suppression(lefteye_coordinates[0], lefteye_scores[0], distance_threshold)
        best_righteye_index = self.non_maximum_suppression(righteye_coordinates[0], righteye_scores[0], distance_threshold)
        best_nose_index = self.non_maximum_suppression(nose_coordinates[0], nose_scores[0], distance_threshold)
        best_leftmouth_index = self.non_maximum_suppression(leftmouth_coordinates[0], leftmouth_scores[0], distance_threshold)
        best_rightmouth_index = self.non_maximum_suppression(rightmouth_coordinates[0], rightmouth_scores[0], distance_threshold)

        # 獲取最佳預測坐標
        best_lefteye_coordinate = lefteye_coordinates[0, best_lefteye_index]
        best_righteye_coordinate = righteye_coordinates[0, best_righteye_index]
        best_nose_coordinate = nose_coordinates[0, best_nose_index]
        best_leftmouth_coordinate = leftmouth_coordinates[0, best_leftmouth_index]
        best_rightmouth_coordinate = rightmouth_coordinates[0, best_rightmouth_index]

        # 計算均值作為最終預測結果
        return {
            'lefteye': best_lefteye_coordinate.mean(dim=0),
            'righteye': best_righteye_coordinate.mean(dim=0),
            'nose': best_nose_coordinate.mean(dim=0),
            'leftmouth': best_leftmouth_coordinate.mean(dim=0),
            'rightmouth': best_rightmouth_coordinate.mean(dim=0)
        }
    
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=pretrain['loss']['alpha'] , beta=pretrain['loss']['beta'], distance_threshold_ratio=0.1):
        """
        損失函數函式封裝 class

        Args:
            alpha (float): 權重, 座標位置 Loss 對總 Loss 的占比
            beta (float): 權重, 分類 Loss 對總 Loss 的占比
            distance_threshold_ratio (float): 多少「座標點」占比為正樣本, 默認為 0.25
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.distance_threshold_ratio = distance_threshold_ratio

        self.location_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def get_positive_samples_and_classification_tensor(self, locations_pred, locations_true):
        """
        判斷正樣本和負樣本

        Args:
            locations_pred (tensor): 一堆預測點, 形狀為 (batch_size=1, n, 2) 的張量
            locations_true (tensor): ground truth 座標點, 形狀為 (batch_size=1, 8) 的張量
        Returns:
            positive_indices (list of list): 每個 ground truth 座標點的正樣本索引 list
            min_classification_labels (tensor): 每個預測點的最可能標籤
        """

        batch_size, num_of_point_set_of_predict, _ = locations_pred.size()

        # 將 locations_true 改變形狀為 (batch_size=1, 4, 2), 4 組座標點
        locations_true = locations_true.view(batch_size, 4, 2)

        # 計算每個預測點與所有 ground truth 點的距離 ( cdist 函數指定 p=2 即為歐幾理得距離 )
        """
        到這一步得到的是
        [[[ d1,  d2,  d3,  d4 ],
          [ d5,  d6,  d7,  d8 ], 
          [ d9, d10, d11, d12 ],
          ......]]
        
        橫排 ( d1~d4 ) 代表一個預測座標對 4 個 true 座標的距離
        直排 ( d1、d5、d9... ) 代表所有預測座標到第一個 true 座標的距離
        """
        distances = torch.cdist(locations_pred, locations_true, p=2)  # (batch_size=1, n, 4)
        # print(distances)

        # 用於儲存對於每個 ground truth 座標點來說, 為正樣本的預測點 idx list
        positive_idx_list = [[] for _ in range(4)]

        # 針對每個標籤點計算最小距離的前 distance_threshold_ratio 比例的預測點
        for label_idx in range( 4 ):
            # 依照 label_idx 取得所有預測標點座標點對於每個 ground truth 座標點的距離
            single_true_label_distances = distances[:, :, label_idx]  # (batch_size, n)
            # 依照 distance_threshold_ratio * 所有預測座標點的總數, 取得需要獲取的前 k 個距離
            k = int( self.distance_threshold_ratio * num_of_point_set_of_predict )
            # 使用 topk 方法由小到大排序後, 挑出前 k 個距離最小的值, 並從其中取最大的那個做為 distance_threshold
            distance_threshold = single_true_label_distances.topk( k, largest=False )[0].max()

            # 從對單一 ground truth 的所有預測座標距離中, 選出那些小於 distance_threshold 的 idx 並記錄
            current_positive_indices = single_true_label_distances <= distance_threshold
            ### 將 current_positive_indices 記錄到 positive_indices 中
            indices = current_positive_indices[0].nonzero(as_tuple=False).squeeze()
            # 針對「只有一個值」或「有多個值」做不同處理
            if indices.ndim == 0:  # 如果只有一個值
                indices = [indices.item()]
            else:
                indices = indices.tolist()
            positive_idx_list[label_idx] = indices

        # 儲存每個預測點 ( 對所有的類別標籤來說 ) 的最小距離, 其大小和傳入的預測點數量 n 相同, 且初始值為無限大 ( inf )
        min_distance = torch.full((num_of_point_set_of_predict,), float('inf'), dtype=torch.float32)
        # 儲存每個預測點的儲存類別標籤, 其大小和傳入的預測點數量 n 相同, 且初始值為 -1 ( 表示沒有類別, 即不滿足 ratio )
        min_classification_labels = torch.full((num_of_point_set_of_predict,), -1, dtype=torch.int32)

        # 遍歷每個標籤
        for label_idx in range(4):
            # 遍歷每個標籤內的正樣本索引 list 的 idx
            for idx in positive_idx_list[label_idx]:
                # 取得預測點的實際距離的值
                distance = distances[0, idx, label_idx].item()
                # 如果這個預測點的最小距離小於已經記錄過的最小距離 ( 這過程包含初次記錄, 反之如果不是初次記錄, 會比較新預測點和上次的預測點誰的距離更小, 來獲得類別標籤 )
                if distance < min_distance[idx]:
                    # 則把該預測點的最小距離修正為這個
                    min_distance[idx] = distance
                    # 並且讓對應的標籤類別更新
                    min_classification_labels[idx] = label_idx

        # 建立最終的 positive_idx_list
        final_positive_idx_list = [[] for _ in range(4)]
        for idx in range(num_of_point_set_of_predict):
            # 遍歷每個預測點的最小距離標籤
            label_idx = min_classification_labels[idx].item()
            # 如果標籤值不為 -1, 就是我們要的標籤值
            if label_idx != -1:
                final_positive_idx_list[label_idx].append(idx)

        # print(min_classification_labels)

        return final_positive_idx_list, min_classification_labels

    def forward(self, locations_pred, classifications_pred, locations_true, image_size):
        """
        計算總損失

        Args:
            locations_pred (tensor): 預測的座標位置, 形狀為 (batch_size=1, n, 2)
            classifications_pred (tensor): 預測的分類結果, 形狀為 (batch_size=1, n, num_classes)
            locations_true (tensor): ground truth 座標位置, 形狀為 (batch_size=1, 8)
            image_size (tuple): 圖片的大小 (height, width)

        Returns:
            total_loss (tensor): 總損失值
        """
        # 分配樣本, 得到「正樣本索引」和每個預測點的對應類別的「tensor」
        positive_idx_list, classification_labels = self.get_positive_samples_and_classification_tensor( locations_pred , locations_true )

        # 用於儲存位置損失與分類損失
        location_loss = 0.0
        classification_loss = 0.0

        # 轉換 input label 從 1*8 為 4*2
        locations_true = locations_true.view(4, 2)
        # 縮放座標
        #print(f"locations_pred:\n{locations_pred}")
        #print(f"locations_true:\n{locations_true}")

        height, width = image_size
        # 取得縮放大小, 並將張量移動到與其他張量相同的 device 上
        size_tensor = torch.tensor([width, height], device=locations_pred.device)
        locations_pred = torch.clamp(locations_pred / size_tensor, 0, 1)
        locations_true = torch.clamp(locations_true / size_tensor, 0, 1)

        #print(f"locations_pred:\n{locations_pred}")
        #print(f"locations_true:\n{locations_true}")

        # 計算座標位置損失
        for label_idx, location_idx in enumerate( positive_idx_list ):
            if location_idx:
                pos_pred = locations_pred[ 0, location_idx, :]
                pos_true = locations_true[label_idx, :].expand_as(pos_pred)
                # print(f"pos_pred:\n{pos_pred}")
                # print(f"pos_true:\n{pos_true}")
                single_location_loss = self.location_loss_fn( pos_pred, pos_true )
                print(f"location loss {label_idx}               : {single_location_loss:04.4f} * {self.alpha} = {single_location_loss*self.alpha:04.4f}")
                location_loss += single_location_loss
                # print("--------------------")

        ### 處理背景類別 ( 的位置損失: 不計算損失 )
        # 取得背景類別的索引
        background_class_idx = len( positive_idx_list ) + 1 - 1 
        # 取得類別為 -1 的 indices
        background_indices = (classification_labels == -1)

        # 計算出正樣本數總共有多少
        non_background_sample_count = (classification_labels != -1).sum().item()
        # 給定最大背景樣本的數量 ( 不然背景部份總量會過於龐大 )
        max_background_sample_count = int( non_background_sample_count * pretrain['loss']['ratio_non_background'] )

        # 降背景樣本數量
        if background_indices.sum().item() > max_background_sample_count:
            background_indices = torch.multinomial(background_indices.float(), max_background_sample_count, replacement=False).tolist()
        else:
            background_indices = torch.nonzero(background_indices, as_tuple=False).squeeze().tolist()

        # 計算背景類別的分類損失
        if background_indices:
            # 取得為「背景」的預測類別結果, tensor 形狀為 (n, 5)
            background_class_pred = classifications_pred[0, background_indices, :]
            # 為背景類別添加等同長度的 ground truth tensor, 背景類別即為 background_class_idx, 因為為最後一個類別
            background_class_true = torch.full( (len(background_indices),) , background_class_idx, dtype=torch.long, device=background_class_pred.device)
            background_classification_loss = self.classification_loss_fn( background_class_pred , background_class_true )
            print(f"background classification loss: {background_classification_loss:04.4f} * {self.beta} = {background_classification_loss*self.beta:04.4f}")
            classification_loss += background_classification_loss

        # 計算正樣本的分類損失
        for label_idx, location_idx in enumerate( positive_idx_list ):
            if location_idx:
                class_pred = classifications_pred[0, location_idx, :]
                class_true = torch.full( ( len(location_idx), ), label_idx, dtype=torch.long, device=class_pred.device ) # 真實類別即為 label_idx, 因為 label 檔案室依照左眼、右眼、鼻子、嘴巴的順序排序的
                # print(f"class_pred:\n{class_pred}")
                # print(f"class_true:\n{class_true}")
                single_classification_loss = self.classification_loss_fn( class_pred , class_true ) # 交叉熵損失內部會自動調用 softmax
                print(f"classification loss {label_idx}         : {single_classification_loss:04.4f} * {self.beta} = {single_classification_loss*self.beta:04.4f}")
                classification_loss += single_classification_loss
                # print("--------------------")

        print("====================")
        # 合併損失
        total_loss = self.alpha * location_loss + self.beta * classification_loss
        return total_loss
    
class MultiTaskDecoder(nn.Module):
    def __init__(self, confidence_threshold=0.5, top_k=1, nms_distance_threshold=20):
        """
        解碼器類別，用於解析模型的預測結果

        Args:
            confidence_threshold (float): 置信度閾值, 默認為 0.4
            top_k (int): 每類別選擇的最大預測數量, 默認為 1
            nms_distance_threshold (float): Non-Maximum Suppression (NMS) 的閾值, 座標之間彼此超過多少像素距離才會保留, 默認為 20
        """
        super(MultiTaskDecoder, self).__init__()
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_distance_threshold = nms_distance_threshold

    def forward(self, locations, classifications):
        """
        解碼預測結果

        Args:
            locations (tensor): 預測的座標位置 (batch_size=1, n, 2)
            classifications (tensor): 預測的分類結果 (batch_size=1, n, num_classes)
            image_size (tuple): 圖片的大小 (height, width)

        Returns:
            output (list): 解碼後的預測結果, 每個batch一個list
        """
        batch_size, num_points, _ = locations.size()
        # 用於儲存最終的預測座標們
        output = []

        for i in range(batch_size):
            points = locations[i]  # 取出第 i 個 batch 的所有預測點
            scores = torch.softmax(classifications[i], dim=-1)  # 對分類結果應用 softmax, 得到所有預測點對所有概率的信心分數, (n, num_classes)
            results = []

            for class_idx in range( scores.size(1) ):
                class_confidence_scores = scores[:, class_idx]  # 取出第 class_idx 類別的信心分數
                mask = class_confidence_scores > self.confidence_threshold  # 製作篩選遮罩, 只保留大於置信度閾值的預測點
                if mask.sum() == 0:
                    continue
                class_points = points[mask]                             # 使用遮罩, 只保留大於置信度閾值的錨點
                class_confidence_scores = class_confidence_scores[mask] # 使用遮罩, 只保留大於置信度閾值的信心分數

                # 使用被保留的錨點與其信心分數進行 NMS, 取得要被保留的 idx, 做成 keep 遮罩
                keep = self.nms(class_points, class_confidence_scores)
                # 利用 keep 遮罩保留需要保留的錨點與信心分數
                class_points = class_points[keep]
                class_confidence_scores = class_confidence_scores[keep]

                # 利用 top_k 參數來限制每個類別返回的座標數量
                if len( class_confidence_scores ) > self.top_k:
                    top_k_scores, top_k_indices = class_confidence_scores.topk(self.top_k, largest=True, sorted=True)
                    class_points = class_points[top_k_indices]
                    class_confidence_scores = top_k_scores

                for point, score in zip(class_points, class_confidence_scores):
                    results.append((class_idx, score, point))

            output.append(results)

        return output

    def nms(self, locations, scores):
        """
        Non-Maximum Suppression (NMS) 非最大抑制方法

        Args:
            locations (tensor): 預測的錨點 (num_locations, 2)
            scores (tensor): 預測類別的信心值 (num_locations,)

        Returns:
            keep (tensor): 保留的 index
        """

        # 儲存需要保留的 idx list
        keep_idx_list = []
        # 取得排序後的「原序列」idx
        _, idxs = scores.sort(0, descending=True)
        """
        1. numel 函數會檢查待處理的張量還有多少元素在其
        2. while 依照以下邏輯進行篩選:
            - 以信心度最高者當作第一個真正的座標點
            - 篩選掉距離小於 threshold 的座標後
            - 如此循環, 以下一個信心度最高的座標點當作第二個真正的座標點...
        """
        while idxs.numel() > 0:
            # 獲取當前最大信心度的原始序列 idx
            this_idx = idxs[0]
            # 儲存當前最大信心度的原始序列 idx
            keep_idx_list.append(this_idx)
            # 如果 idxs 裡面只有一個元素, 那麼也沒有其他點可以排除, 因此退出循環
            if idxs.numel() == 1:
                break
            # 計算當前最大信心度的座標與其他剩餘座標的距離
            distances = self.euclidean_distance( locations[this_idx] , locations[idxs[1:]])
            # 取出非當前的 idx, 並使用 bool list comprehension 將距離小於 threshold 的 idx 從 idxs 中移除
            idxs = idxs[1:][distances > self.nms_distance_threshold]
            # 開始下一個循環

        return torch.tensor(keep_idx_list, dtype=torch.long)

    def euclidean_distance(self, point1, points2):
        """
        計算歐幾里得距離

        Args:
            point1 (tensor): 單一預測點 (2,)
            points2 (tensor): 多個預測點 (num_boxes, 2)

        Returns:
            distances (tensor): 歐幾里得距離值 (num_boxes,)
        """
        return torch.norm(points2 - point1, dim=1)
    
