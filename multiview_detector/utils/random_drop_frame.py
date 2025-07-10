import torch
import random

def random_drop_frame(features, num_cam, tau_1, channel, drop_prob):
    B, C, H, W = features.shape  # C = self.channel * self.num_cam * (self.tau_1 + 1)
    
    # 掩码形状为 (B, num_cam * (tau_1 + 1))
    mask = torch.ones(B, num_cam * (tau_1 + 1), dtype=torch.float32, device=features.device)

    # 随机丢弃相机的某些帧
    for b in range(B):
        for c in range(num_cam):
            if random.random() < drop_prob:
                # 将某个相机的所有帧标记为 0，丢弃它
                mask[b, c*(tau_1 + 1):(c+1)*(tau_1 + 1)] = 0

    # 保存原始掩码，用于后续计算
    original_mask = mask.clone()

    # 将掩码扩展到与 features 匹配的通道数，并调整维度
    mask = mask.unsqueeze(2).repeat(1, 1, channel).view(B, C, 1, 1).expand(-1, -1, H, W) # mask的形状为 (B, self.num_cam * (self.tau_1 + 1))

    # 应用掩码
    masked_features = features * mask

    return masked_features, original_mask

import pandas as pd
import torch
import os
import random


def random_drop_frame_with_priority(features, num_cam, tau_1, channel, target_dropout_rate, is_training=True):
    B, C, H, W = features.shape  # C = self.channel * self.num_cam * (tau_1 + 1)

    # 打印 features 的维度，调试用
    # print(f"Features shape: {features.shape}")

    # 初始化掩码，形状为 (B, num_cam * (tau_1 + 1))
    mask = torch.ones(B, num_cam * (tau_1 + 1), dtype=torch.float32, device=features.device)

    # 创建一个空的 DataFrame 来存储丢包数据
    dropout_data = {
        'Batch': [],
        'Camera': [],
        'Tau': [],
        'Dropout_Probability': []
    }

    # 确保保存路径存在
    save_dir = "/data1/fangzr/Research/24-JSAC-EC-multiview/RTFS/temp/feature_temp"
    os.makedirs(save_dir, exist_ok=True)

    # 遍历所有相机帧，并根据特征的低值优先保留
    for b in range(B):
        total_frames = num_cam * (tau_1 + 1)  # 总帧数
        current_dropout_count = 0  # 当前已丢包的帧数
        
        # 计算目标丢包帧数
        target_dropout_count = int(total_frames * target_dropout_rate)

        # 打印当前 features[b] 的形状
        # print(f"Batch {b} features shape: {features[b].shape}")

        # 使用特征的平均值来表示优先级，数值越低的特征越重要
        if features[b].dim() == 3:  # 确保 features[b] 是 [C, H, W]
            avg_feature_values = features[b].mean(dim=[1, 2])  # 在 H 和 W 维度求平均
            # print(f"Avg feature values for Batch {b}: {avg_feature_values}")
        else:
            raise ValueError(f"Unexpected feature shape: {features[b].shape}")

        # 构建包含 camera 和 tau 的索引列表
        camera_tau_list = [(cam_idx, tau) for cam_idx in range(num_cam) for tau in range(tau_1 + 1)]
        priority_values_with_indices = [(avg_feature_values[i], camera_tau_list[i]) for i in range(total_frames)]
        
        # 按优先级进行排序 (低优先级在前)
        priority_sorted = sorted(priority_values_with_indices, key=lambda x: x[0])

        # 动态计算丢包概率
        min_priority = min([x[0] for x in priority_sorted]).item()
        max_priority = max([x[0] for x in priority_sorted]).item()

        # # 针对每个 tau 打印同一 tau 下的相机优先级
        # if not is_training:
        #     for tau in range(tau_1 + 1):
        #         print(f"Batch {b}, Tau {tau}, Priority Sorting for Cameras:")
        #         for priority_value, (cam_idx, tau_val) in priority_sorted:
        #             if tau_val == tau:  # 仅打印相同 tau 的 camera
        #                 print(f"Camera {cam_idx}: Priority Value {priority_value.item()}")
                        
        # 遍历所有相机帧，按优先级决定是否丢包
        for priority_value, (cam_idx, tau) in priority_sorted:
            if current_dropout_count >= target_dropout_count:
                break  # 达到目标丢包率，停止丢包

            # 计算 mask 索引
            i = cam_idx * (tau_1 + 1) + tau

            # 根据优先级动态计算丢包概率，数值越高丢包概率越大
            drop_prob = (priority_value.item() - min_priority) / (max_priority - min_priority + 1e-6)  # 正则化到 0 到 1
            

            # 如果随机数小于丢包概率，则丢弃
            if random.random() < drop_prob:
                mask[b, i] = 0  # 丢弃该相机帧
                current_dropout_count += 1

                # 打印丢包信息（仅在测试阶段）
                if not is_training:
                    # 保存特征的 HxW
                    feature_slice = features[b, i*channel:(i+1)*channel, :, :]  # 获取相应相机的特征
                    feature_filename = f"batch_{b}_camera_{cam_idx}_tau_{tau}.pt"
                    torch.save(feature_slice.cpu(), os.path.join(save_dir, feature_filename))

                    # 将信息存入 Excel 数据
                    dropout_data['Batch'].append(b)
                    dropout_data['Camera'].append(cam_idx)
                    dropout_data['Tau'].append(tau)
                    dropout_data['Dropout_Probability'].append(drop_prob)

    # 将掩码扩展到与 features 匹配的通道数，并调整维度
    expanded_mask = mask.unsqueeze(2).repeat(1, 1, channel).view(B, C, 1, 1).expand(-1, -1, H, W)  # 形状为 (B, C, H, W)

    # 应用掩码
    masked_features = features * expanded_mask

    # 将丢包数据存入 Excel 文件
    if not is_training:
        df = pd.DataFrame(dropout_data)
        excel_path = os.path.join(save_dir, 'dropout_data.xlsx')
        df.to_excel(excel_path, index=False)

    return masked_features, mask  # 返回原始掩码而不是扩展后的掩码


def test_random_drop_frame():
    # 模拟输入
    B = 1  # 批量大小
    num_cam = 7  # 相机数量
    tau_1 = 2  # 时间步数量
    channel = 8  # 每个相机的特征通道数
    H, W = 120, 360  # 特征图的空间尺寸
    
    # 随机生成特征，形状为 (B, self.channel * self.num_cam * (self.tau_1 + 1), H, W)
    features = torch.randn(B, channel * num_cam * (tau_1 + 1), H, W).cuda()  # 在GPU上创建张量

    # 调用 random_drop_frame 函数
    masked_features, original_mask = random_drop_frame(features, num_cam, tau_1, channel, drop_prob=0.2)

    # 打印特征和掩码的形状
    print("输入特征形状: ", features.shape)
    print("掩码形状: ", original_mask.shape)  # 打印原始掩码形状
    print("掩码后的特征形状: ", masked_features.shape)

    # 查看掩码的内容，检查随机丢弃是否生效
    print("掩码: ", original_mask)

    # 检查丢弃的帧数据是否为0
    for b in range(B):
        for c in range(num_cam):
            for t in range(tau_1 + 1):
                mask_value = original_mask[b, c * (tau_1 + 1) + t].item()
                if mask_value == 0:  # 该帧应该被丢弃
                    print(f"Camera {c}, time {t}, batch {b} has been dropped.")
                    # 检查该相机的所有通道在这一帧是否都被置为0
                    start_channel = (c * (tau_1 + 1) + t) * channel
                    end_channel = start_channel + channel
                    feature_slice = masked_features[b, start_channel:end_channel, :, :]
                    assert torch.all(feature_slice == 0), f"Camera {c}, time {t}, batch {b} did not mask correctly!"

    print("所有丢弃的帧数据都已成功置为0。")



# 运行测试
if __name__ == "__main__":
    test_random_drop_frame()
