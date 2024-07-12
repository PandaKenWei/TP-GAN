import torch
from MobileNetV2 import MultiTaskLoss, MultiTaskDecoder

# 假設的 distance_threshold_ratio
distance_threshold_ratio = 0.5

# 假設的預測座標點 (batch_size=1, n=12, 2)
locations_pred = torch.tensor([[[  1.0,   1.0], [420.0, 360.0], [370.0, 150.0], [180.0, 220.0],
                                [330.0, 270.0], [290.0, 135.0], [500.0, 380.0], [190.0, 400.0], 
                                [210.0, 420.0], [510.0,  70.0], [178.0, 321.0], [420.0, 110.0]]])

# 假設的 ground truth 座標點 (batch_size=1, 8)
locations_true = torch.tensor([[0.0, 0.0, 150.0, 400.0, 350.0, 250.0, 300.0, 150.0]])

# 假設的分類結果 (batch_size=1, n=12, num_classes=3)
classifications_pred = torch.tensor([[[2.0, 1.0, 0.1, 0.5, 1.4], [1.0, 2.0, 0.1, 0.3, 1.1], [0.1, 2.0, 1.0, 0.4, 0.5],
                                      [2.0, 0.1, 1.0, 0.7, 0.5], [1.0, 0.1, 1.4, 0.8, 2.0], [0.1, 1.0, 2.0, 0.6, 0.7],
                                      [2.0, 1.0, 0.1, 0.9, 1.5], [1.0, 0.8, 0.1, 1.1, 2.0], [0.1, 1.2, 1.0, 2.0, 0.5],
                                      [2.0, 0.1, 1.0, 1.3, 0.6], [1.0, 0.1, 2.0, 1.4, 1.6], [0.1, 1.0, 1.3, 1.5, 2.0]]])

loss_fn = MultiTaskLoss()
decoder = MultiTaskDecoder(nms_distance_threshold=30)

# 計算總損失
total_loss = loss_fn( locations_pred , classifications_pred , locations_true , (600, 800))

print(f"Total Loss: {total_loss.item()}")

# points = torch.tensor([
#     [0.1, 0.2],
#     [0.15, 0.25],
#     [0.2, 0.3],
#     [0.8, 0.8],
#     [0.9, 0.9],
#     [0.95, 0.95],
#     [0.5, 0.5],
#     [0.55, 0.55],
#     [0.6, 0.6],
#     [0.65, 0.65]
# ])

# print(points)

# scores = torch.tensor([0.9, 0.85, 0.95, 0.6, 0.75, 0.5, 0.3, 0.4, 0.2, 0.1])

# keep_indices = decoder.nms(points, scores)

# print("保留的錨點索引：", keep_indices)
# print("保留的錨點：", points[keep_indices])
# print("保留的分數：", scores[keep_indices])

print(f"假的預測座標:\n{locations_pred}")
print(f"假的預測類別:\n{classifications_pred}")

decoded_output = decoder( locations_pred, classifications_pred)[0]


for class_label, score_tensor, point_tensor in decoded_output:
    print(f"預測類別: {class_label}")
    print(f"預測信心: {score_tensor.item():.4f}")
    print(f"預測座標: ({point_tensor[0].item():.4f}, {point_tensor[1].item():.4f})")
    print("---------------------")