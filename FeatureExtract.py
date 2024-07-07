from ResNet import ResNet18
from MobileNetV2 import MobileNetV2
import torch.nn as nn

class FeatureExtractModel(nn.Module):
    def __init__(self, base_model_name='resnet', num_of_output_classes=1000, use_pretrained=False, **kwargs):
        """
        初始化特徵提取模型，可以選擇使用 ResNet 或 MobileNetV2 作為基礎模型。

        Args:
            base_model_name (str): 選擇基礎模型,「resnet」或「mobilenetv2」, 默認為「resnet」
            num_of_output_classes (int): 最終分類的類別數量。
            use_pretrained (bool): 是否使用預訓練的模型權重。
            **kwargs: 傳遞給基礎模型的其他參數。
        """
        super(FeatureExtractModel, self).__init__()
        # 確認傳入模型名稱全為小寫
        self.base_model_name = base_model_name.lower()
        # 給定預測的最終類別數
        self.num_of_output_classes = num_of_output_classes

        if self.base_model_name == 'resnet':
            self.base_model = ResNet18(**kwargs)
        elif self.base_model_name == 'mobilenetv2':
            self.base_model = MobileNetV2(**kwargs)
        else:
            raise ValueError("特徵提取模型僅支持 ResNet18 或 MobilNetV2")

        # 替換最終的全連接層，根據需要進行分類
        if self.base_model_name == 'resnet':
            in_features = self.base_model.FC.in_features
            self.base_model.FC = nn.Linear(in_features, num_of_output_classes)
        elif self.base_model_name == 'mobilenetv2':
            in_features = self.base_model.FC[-1].in_features
            self.base_model.FC = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_of_output_classes),
            )

    def forward(self, x):
        return self.base_model(x)