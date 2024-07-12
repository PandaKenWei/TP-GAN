##### Pretrain 參數 #####

pretrain = {}
pretrain['txt_name'] = 'list_landmarks_celeba.txt'                            # pretrain 資料集的 label 檔案
pretrain['data_root_dir'] = 'C:\\Users\\User\\Downloads\\CelebA'              # pretrain 資料集的根目錄路徑
pretrain['log_root_dir'] = 'C:\\Users\\User\\Desktop\\Test\\SummaryWriterLog' # SummaryWriter 的 Log 的放置處的根目錄路徑
pretrain['model_name'] = 'MobileNetV2'                                        # pretrain 所使用的模型名稱

pretrain['train_data_ratio'] = 0.95        # 訓練資料集的比例
pretrain['validation_data_ratio'] = 0.0005 # 驗證資料集的比例
# 不設定測試集的比例是因為要「用總數直接減去上兩者」, 不然可能會發生三部分資料集加總不等於原始資料集數量的問題
pretrain['batch_size'] = 1                 # batch size 設為 1 才能處理輸入空間尺寸不同的圖片

pretrain['optimizer'] = 'SGD'

pretrain['use_learning_rate_scheduler'] = True               # 是否使用學習率調度器
pretrain['learning_rate_scheduler_milestone'] = [10, 20, 30] # 指定個 epoch 降低學習率
pretrain['learning_rate_scheduler_gamma'] =  0.1             # 每次學習率降低會變為原來的 n 倍

pretrain['num_epochs'] = 5          # 指定訓練的 epochs 數
pretrain['log_step_of_batchs'] = 200 # 記錄一次 log 所需的 batch 數

##### Loss 參數 #####
pretrain['loss'] = {}
pretrain['loss']['alpha'] = 30.0
pretrain['loss']['beta'] = 0.1
pretrain['loss']['ratio_non_background'] = 5.0


##### Optimizer 參數 #####
optimizer_param = {}
optimizer_param['learning_rate'] = 5e-4  # 學習率
optimizer_param['momentum'] = 0.9        # 動量係數
optimizer_param['nesterov'] = True       # 是否使用 Nesterov 加速
optimizer_param['weight_decay'] = 5e-4   # L2 正則化係數


##### 通用參數 #####
general = {}
general['image_max_size'] = 1024         # 圖片單邊解析度最大像素







####### 以下不確定
# 訓練參數
train =  {}
train['img_list'] = './img.list' # 訓練資料路徑
train['learning_rate'] = 1e-4    # learning rate
train['num_epochs'] = 50         # 訓練週期總數
train['batch_size'] = 50         # batch size
train['log_step'] = 1000         # ???
train['resume_model'] = None     # ???
train['resume_optimizer'] = None # ???

# 生成器參數
G = {}
G['zdim'] = 64                   # 輸入的隨機燥聲維度
G['use_residual_block'] = False  # 是否使用「殘差連接區塊」
G['use_batchnorm'] = False       # 是否使用「批量正規化」
G['num_classes'] = 347           # 分類數量

# 鑑別器參數
D = {}
D['use_batchnorm'] = False       # 是否使用「批量正規化」

# 損失函數權重
loss = {}
loss['weight_gradient_penalty'] = 10       # 使用 Wasserstein GANs（WGAN）損失的梯度懲罰權重
loss['weight_128'] = 1.0                   # 不同尺寸的特徵圖的權重
loss['weight_64'] = 1.0
loss['weight_32'] = 1.5
loss['weight_pixelwise'] = 1.0             # 像素級別的圖片的損失的權重
loss['weight_pixelwise_local'] = 3.0       # 局部像素級別的圖片的損失的權重
loss['weight_symmetry'] = 3e-1             # 對稱性損失的權重
loss['weight_adv_G'] = 1e-3                # 生成器對抗損失的權重
loss['weight_identity_preserving'] = 3e1   # 保持身份 ( 風格 ) 一致性損失權重
loss['weight_total_varation'] = 1e-3       # 總異變損失的權重
loss['weight_cross_entropy'] = 1e1         # 交叉熵損失的權重
# 特徵提取模型的參數
feature_extract_model = {}
feature_extract_model['resume'] =  'save/feature_extract_model/resnet18/try_1' # 特徵提取的模型檔案的路徑