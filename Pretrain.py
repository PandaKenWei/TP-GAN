import os
import copy
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

from config import pretrain
from MobileNetV2 import MobileNetV2
from DataAndDataset import PretrainDataset
from UtilityMethods import getOptimizer, save_model, save_optimizer

def _calculate_loss( predicts, ground_truth ):
    """
    分別計算四個部位的均方誤差並加總

    Args:
        predicts (Tensor): 形狀為 [batch_size, 8] 的預測座標張量
        ground_truth (Tensor): 形狀為 [batch_size, 8] 的真實座標張量

    Returns:
        Tensor: 均方誤差損失
    """

    # 確保 predicts 和 ground_truth 具有相同的形狀
    if predicts.shape != ground_truth.shape:
        raise ValueError("Predicts and ground truth tensors must have the same shape")

    # 計算均方誤差損失
    loss = F.mse_loss( predicts , ground_truth , reduction='mean')
    
    return loss

def _calculate_accuracy(predicts, ground_truth):
    """
    計算預測與真實座標之間的平均絕對誤差作為準確度

    Args:
        predicts (Tensor): 形狀為 [batch_size, 8] 的預測座標張量
        ground_truth (Tensor): 形狀為 [batch_size, 8] 的真實座標張量

    Returns:
        float: 平均絕對誤差
    """

    # 確保 predicts 和 ground_truth 具有相同的形狀
    if predicts.shape != ground_truth.shape:
        raise ValueError("Predicts and ground truth tensors must have the same shape")

    # 計算平均絕對誤差
    absolute_errors = torch.abs(predicts - ground_truth)
    mean_absolute_error = torch.mean(absolute_errors).item()
    
    return mean_absolute_error

if __name__ == "__main__":
    ############### 可視化套件的部分 ###############
    # 創建 SummaryWriter 實例
    log_dir = os.path.join( pretrain['log_root_dir'], pretrain['model_name'])
    writer = SummaryWriter( log_dir=log_dir )

    # 創建日誌文件
    log_file_path = os.path.join( writer.log_dir, 'train', 'log.txt')
    os.makedirs( os.path.dirname( log_file_path ), exist_ok=True )
    log_file = open( log_file_path, 'w' )

    # 取得 Pretrain 資料集的根目錄路徑與 label 檔案名稱
    data_root_dir = pretrain['data_root_dir'] # 資料集根目錄
    txt_name = pretrain['txt_name'] # label 檔案名稱

    ############### 資料集的準備 ###############
    # 創建 PretrainDataset 實例
    dataset = PretrainDataset( txt_name, data_root_dir )
    
    # 給定比例並計算 train、validation、test 三部分的資料大小
    total_record_of_dataset = len( dataset )
    train_record_of_dataset = int( total_record_of_dataset * pretrain['train_data_ratio'] )
    validation_record_of_dataset = int( total_record_of_dataset * pretrain['validation_data_ratio'] )
    test_record_of_dataset = int( total_record_of_dataset - train_record_of_dataset - validation_record_of_dataset )

    # 使用 random_split 隨機拆分數據集為 3 部份
    train_dataset, validation_dataset, test_dataset = random_split( dataset, [train_record_of_dataset, validation_record_of_dataset, test_record_of_dataset])

    # 為 3 部分數據集創建 DataLoader
    train_loader = DataLoader( train_dataset, batch_size=pretrain['batch_size'], shuffle=True)
    validation_loader = DataLoader( validation_dataset, pretrain['batch_size'], shuffle=True)
    test_loader = DataLoader( test_dataset, pretrain['batch_size'], shuffle=True)

    ############### 模型的準備 ###############
    # 定義模型, 並將其移至 CPU 或 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eval( pretrain['model_name'] )()
    model.to( device )

    # Optimizer
    optimizer = getOptimizer( model.parameters(), pretrain['optimizer'] )()

    # 檢查是否使用學習率調度器
    if pretrain.get('use_learning_rate_scheduler', False):
        # 初始化學習率調度器
        learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, 
                                                                        milestones=pretrain['learning_rate_scheduler_milestone'], 
                                                                        gamma=pretrain['learning_rate_scheduler_gamma'])

    # 定義損失函數
    loss_function = torch.nn.SmoothL1Loss()

    ############### 一些變數設置 ###############
    # 開始訓練前, 記錄當下時間
    now_time = time.time()

    # 用於儲存每個 epoch 的 loss 和 accuracy 的 list
    training_loss_every_epoch_list = []
    training_accuracy_every_epoch_list = []
    # 用於儲存每個 epoch 中的每一個 batch 出來的 accuracy 和 loss 的 list
    training_accuracy_every_batch_in_epoch_list = []
    training_loss_every_batch_in_epoch_list = []
    # 用於儲存每次驗證集的 accuracy 和 loss 的 list
    val_accuracy_every_specific_step_list = []
    val_loss_every_specific_step_list = []

    ############### 訓練環節 ###############
    for epoch in range( pretrain['num_epochs'] ):
        ### 初始化一些變數 ### 
        best_validation_accuracy = 0 # 記錄「最佳驗證準確度」
        best_model = None            # 記錄「最佳模型」

        # 單個 epoch 中的批次訓練, 遍歷數據集中的每一個批次
        # 這邊 images 和 labels 都是一整個批次的數據
        for step, ( images, labels ) in enumerate( train_loader ):
            # 將 batch 中的每一筆 Data、label 都移至「指定設備」
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 設置模型為訓練模式
            model.train()

            # 輸入數據進行座標預側
            eyes_left, eyes_right, nose, mouth = model( images, use_dropout=True )

            # 將預測結果拼接成一個 1*8 張量以便後續計算損失 -> [batch_size, 8]
            predicts = torch.cat( (eyes_left, eyes_right, mouth, nose) , dim=1)

            # 計算損失
            train_loss = _calculate_loss( predicts , labels )

            # 計算準確度
            train_accuracy = _calculate_accuracy( predicts, labels )

            # 反向傳播和參數更新
            optimizer.zero_grad() # 反向傳播前將梯度歸 0, 否則梯度會累積
            train_loss.backward() # 反性傳播
            optimizer.step()      # 更新模型參數

            # 記錄訓練損失和準確度
            training_loss_every_batch_in_epoch_list.append( train_loss.item() )
            training_accuracy_every_batch_in_epoch_list.append( train_accuracy )

            # 每隔一定步數進行驗證
            if step % pretrain['log_step_of_batchs'] == 0:

                # 切換模型為評估模式
                model.eval()

                # 自動禁用梯度區塊
                with torch.no_grad():

                    # 創建一個 list 來存儲驗證集損失和準確度
                    val_loss_list = []
                    val_accuracy_list = []

                    # 遍歷驗證集數據
                    for val_images, val_labels in validation_loader:
                        # 將驗證集的 images 和 labels 移動到設備（CPU 或 GPU）
                        val_images = val_images.to( device, non_blocking=True )
                        val_labels = val_labels.to( device, non_blocking=True )

                        # 設置模型為評估模式並進行預測
                        val_eyes_left, val_eyes_right, val_nose, val_mouth = model( val_images, use_dropout=False)
                        
                        # 將預測結果拼接成一個張量 -> [batch_size, 8]
                        val_predicts = torch.cat( ( val_eyes_left, val_eyes_right, val_nose, val_mouth ) , dim=1)
                        
                        # 計算驗證集的損失和準確度
                        val_loss = _calculate_loss( val_predicts, val_labels )
                        val_accuracy = _calculate_accuracy( val_predicts, val_labels )

                        # 將損失添加到 list 中
                        val_loss_list.append( val_loss.item() )
                        val_accuracy_list.append( val_accuracy )
                    
                    # 計算驗證集損失和準確度的平均值
                    val_loss = torch.mean( torch.tensor( val_loss_list ) )
                    val_accuracy = torch.mean(torch.tensor( val_accuracy_list ))

                    # 記錄每次驗證集的平均損失和平均準確率
                    val_loss_every_specific_step_list.append( val_loss.item() )
                    val_accuracy_every_specific_step_list.append( val_accuracy.item() )

                    # 使用 TensorBoard 記錄驗證集損失
                    writer.add_scalar('loss / validation', val_loss.item(), epoch * len(train_loader) + step)
                    writer.add_scalar('accuracy / validation', val_accuracy.item(), epoch * len(train_loader) + step)
                    
                    # 如果當前驗證集準確度優於之前的最佳準確度, 更新最佳準確度並保存最佳模型
                    if val_accuracy < best_validation_accuracy:
                        best_validation_accuracy = val_accuracy
                        best_model = copy.deepcopy(model)
                    
                    # print 並記錄當前 epoch 的訓練和驗證信息
                    log_msg = (f"epoch {epoch}, step {step} / {len(train_loader) - 1}, "
                               f"train_loss {train_loss.item():.5f}, "
                               f"train_accuracy {train_accuracy:.5f}, "
                               f"val_loss {val_loss.item():.5f}, "
                               f"val_accuracy {val_accuracy:.5f}, "
                               f"{pretrain['log_step_of_batches'] * pretrain['batch_size'] / (time.time() - now_time):.1f} imgs/s"
                               f"===============================================")
                    print( log_msg )
                    log_file.write( log_msg + '\n' )

            log_msg = (f"epoch {epoch}, step {step} / {len(train_loader) - 1}, "
                               f"train_loss {train_loss.item():.5f}, "
                               f"train_accuracy {train_accuracy:.5f}, "
                               f"===============================")
            print( log_msg )

        # 更新學習率調度器 ?????????
        learning_rate_scheduler.step()

        # 可能要打印其他資訊

        # 保存模型和優化器狀態到指定路徑
        save_model(model, writer.log_dir, epoch)
        save_optimizer(optimizer, model, writer.log_dir, epoch)
        print(f"Save done in {writer.log_dir}")

    log_file.close()