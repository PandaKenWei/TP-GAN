import os
import copy
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

from config import pretrain, general
from MobileNetV2 import MobileNetV2, MultiTaskLoss, MultiTaskDecoder
from DataAndDataset import PretrainDataset
from UtilityMethods import getOptimizer, save_model, save_optimizer

def _calculate_accuracy(predicts, ground_truth):
    """
    計算預測與真實座標之間的 euclid distance 作為準確度

    Args:
        predicts (list of tuple): 形狀為 [( 預測類別: int, 預測信心分數: 1*1 tensor, 預測座標: 1*2 tensor )...共 5 個] 的預測座標張量
        ground_truth (Tensor): 形狀為 [batch_size, 8] 的真實座標張量

    Returns:
        float: 經過設計後的 accuracy
    """

    # 1024*1024 -> [10, 20, 31, 41, 51]
    # 96*96     -> [ 1,  2,  3,  4,  5]
    thresholds = [5, 10, 18, 30, 45]
    weights = [1.0, 0.9, 0.65, 0.35, 0.1]

    # 把背景類的預測去除
    predicts = predicts[:-1]
    # 提取 4 組預測座標張量
    predicted_coords = torch.stack([pred[2] for pred in predicts])

    # 將輸入的 1*8 張量轉換成 4*2 的座標張量 (x,y)
    ground_truth = ground_truth.view( -1, 4, 2 )

    # 計算 predict 和 label 座標點之間的歐式距離
    distances = torch.sqrt( torch.sum( (predicted_coords - ground_truth)**2, dim=2 ) )

    # 初始化一個 1*4 的全 0 張量
    accuracy = torch.zeros_like( distances )

    # 定義前一個閥值
    previous_threshold = 0

    # 根據閥值設定加權
    for i, threshold in enumerate( thresholds ):
        weight = weights[i]
        # 製造 mask, EX: [True, False, False, False]
        mask = ( ( distances > previous_threshold ) & ( distances <= threshold ))
        # 將 Mask 的 True 轉換為 1.0, False 轉換為 0.0, 並且乘以權重做為 accuracy
        accuracy += ( mask.float() * weight )
        # 更新前一個閥值, 以免重複加權
        previous_threshold = threshold

    # 計算平均 accuracy
    accuracy = torch.mean( accuracy ).item()

    return accuracy

def  collate_fn( batch, max_size=( general['image_max_size'] , general['image_max_size'] ) ):
    filtered_batch = []
    for item in batch:
        image, label = item
        if image.size(1) <= max_size[0] and image.size(2) <= max_size[1]:
            filtered_batch.append((image, label))
    if len(filtered_batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(filtered_batch)

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
    train_loader = DataLoader( train_dataset, batch_size=pretrain['batch_size'], shuffle=True, collate_fn=lambda x: collate_fn(x))
    validation_loader = DataLoader( validation_dataset, pretrain['batch_size'], shuffle=True, collate_fn=lambda x: collate_fn(x))
    test_loader = DataLoader( test_dataset, pretrain['batch_size'], shuffle=True, collate_fn=lambda x: collate_fn(x))

    ############### 模型的準備 ###############
    # 定義模型, 並將其移至 CPU 或 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = eval( pretrain['model_name'] )()
    model.to( device )

    # Optimizer
    optimizer = getOptimizer( model.parameters(), pretrain['optimizer'] )()

    # loss function
    loss_fn = MultiTaskLoss()

    # decoder
    decoder = MultiTaskDecoder()

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
    print("start training")
    for epoch in range( pretrain['num_epochs'] ):
        ### 初始化一些變數 ### 
        best_validation_accuracy = 0 # 記錄「最佳驗證準確度」
        best_model = None            # 記錄「最佳模型」

        # 單個 epoch 中的批次訓練, 遍歷數據集中的每一個批次
        # 這邊 images 和 labels 都是一整個批次的數據
        for step, batch in enumerate( train_loader ):
            if batch is None:
                continue

            ( images, labels ) = batch
            # 將 batch 中的每一筆 Data、label 都移至「指定設備」
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 設置模型為訓練模式
            model.train()

            # 輸入數據進行座標預側
            locations_pred, classifications_pred = model( images, use_dropout=True )

            # 取得圖片的空間尺寸
            imgage_size = ( images.size(2), images.size(3) )
            # 計算損失
            train_loss = loss_fn( locations_pred , classifications_pred , labels , imgage_size )
            #print(f"step: {step:6} loss: {train_loss.item()}")

            # 計算準確度前先進行 decode
            predicts = decoder( locations_pred , classifications_pred )[0]

            # 計算準確度
            train_accuracy = _calculate_accuracy( predicts , labels )

            # 反向傳播和參數更新
            optimizer.zero_grad() # 反向傳播前將梯度歸 0, 否則梯度會累積
            train_loss.backward() # 反向傳播
            optimizer.step()      # 更新模型參數

            # clear the memory
            torch.cuda.empty_cache()

            # 記錄訓練損失和準確度
            training_loss_every_batch_in_epoch_list.append( train_loss.item() )
            training_accuracy_every_batch_in_epoch_list.append( train_accuracy )

            # 每隔一定步數進行驗證
            if (step+1) % pretrain['log_step_of_batchs'] == 0:

                # 切換模型為評估模式
                model.eval()

                # 自動禁用梯度區塊
                with torch.no_grad():

                    # 創建一個 list 來存儲驗證集損失和準確度
                    val_loss_list = []
                    val_accuracy_list = []

                    # 遍歷驗證集數據
                    for val_batch in validation_loader:

                        if val_batch is None:
                            continue

                        ( val_images, val_labels ) = val_batch
                        # 將驗證集的 images 和 labels 移動到設備（CPU 或 GPU）
                        val_images = val_images.to( device, non_blocking=True )
                        val_labels = val_labels.to( device, non_blocking=True )

                        # 設置模型為評估模式並進行預測
                        val_locations_pred, val_classifications_pred = model( val_images, use_dropout=False)
                        
                        # 取得圖片的空間尺寸
                        val_imgage_size = ( images.size(2), images.size(3) )
                        # 計算驗證集的損失
                        val_loss = loss_fn( val_locations_pred , val_classifications_pred , val_labels , val_imgage_size )

                        # 計算準確度前先進行 decode
                        val_predicts = decoder( val_locations_pred , val_classifications_pred )[0]

                        # 計算驗證集準確度
                        val_accuracy = _calculate_accuracy( val_predicts , val_labels )

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

                    train_mean_loss_in_val = float( sum( training_loss_every_batch_in_epoch_list ) / len( training_loss_every_batch_in_epoch_list ) )
                    train_mean_acc_in_val = float( sum( training_accuracy_every_batch_in_epoch_list ) / len( training_accuracy_every_batch_in_epoch_list ) )
                    
                    # print 並記錄當前 epoch 的訓練和驗證信息
                    log_msg = (
                        f"""===== epoch: {epoch:2}, step: {step+1:6} / {len(train_loader) - 1} =====\n """
                        f"""train_loss: {train_mean_loss_in_val:6.4f}, """
                        f"""train_accuracy: {train_mean_acc_in_val:.4f}\n"""
                        f"""val_loss: {val_loss.item():6.4f}, """
                        f"""val_accuracy {val_accuracy:.4f}\n"""
                        f"""{pretrain['log_step_of_batchs'] * pretrain['batch_size'] / (time.time() - now_time):.1f} imgs/s"""
                        f"""==============================================="""
                    )
                    print( log_msg )
                    log_file.write( log_msg + '\n' )

                    now_time = time.time()

            if (step+1) % 10 == 0:
                #total_memory = torch.cuda.get_device_properties(0).total_memory / 1e6
                #allocated = torch.cuda.memory_allocated() / 1e6
                #reserved = torch.cuda.memory_reserved() / 1e6
                #available_memory = total_memory - reserved
                latest_10_loss = training_loss_every_batch_in_epoch_list[-10:]
                mean_10_loss = float( sum( latest_10_loss ) / len( latest_10_loss ) )
                latest_10_accuracy = training_accuracy_every_batch_in_epoch_list[-10:]
                mean_10_accuracy = float( sum( latest_10_accuracy ) / len( latest_10_accuracy ) )
                log_msg = (
                    f"""epoch: {epoch:2}, step: {step+1:6} / {len(train_loader) - 1} || """
                    f"""mean 10 loss: {mean_10_loss:6.4f}, accuracy: {mean_10_accuracy:.4f}\n"""
                )
                #           f"memory {allocated:.2f} / {reserved:.2f} MB || {available_memory} MB")
                print( log_msg )
                for idx, pred in enumerate( predicts ):
                    if idx < 4:
                        print(f"({pred[2][0]:04.2f}, {pred[2][1]:04.2f} / ({labels[0][ (idx*2) ]}, {labels[0][ (idx*2)+1 ]}))")
                # predict_list = predicts.tolist()[0]
                # predict_str = ' '.join(f"{x:4.4f}" for x in predict_list)
                # label_list = labels.tolist()[0]
                # label_str = ' '.join(f"{x:4.4f}" for x in label_list)
                # print( f"predict: {predict_str}")
                # print( f"label:   {label_str}")
                # print( "--------------------" )

        # 更新學習率調度器 ?????????
        learning_rate_scheduler.step()

        # 可能要打印其他資訊

        # 保存模型和優化器狀態到指定路徑
        save_model(model, writer.log_dir, epoch)
        save_optimizer(optimizer, model, writer.log_dir, epoch)
        print(f"Save done in {writer.log_dir}")

    log_file.close()