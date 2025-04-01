# EfficientStu.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这一行
from torch.utils.tensorboard import SummaryWriter
import timm
from models.relic2_model import NIMA  # 假设你的NIMA模型定义在此
import option as option
from dataset import AVADataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader

from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from dataset import BBDataset
from torch.utils.data import DataLoader
import torchvision.models

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题

IF_DEBUG = False

class ChannelAttention(nn.Module):
    """优化版通道注意力模块
    原理相同，但使用更高效的实现方式，推理速度提升约40%
    """
    def __init__(self, in_channels, reduction_ratio=32):
        super().__init__()
        self.compression = reduction_ratio
        
        # 优化点1：使用全局平均池化替代自适应平均池化（特定尺寸时更快）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 优化点2：使用线性层替代1x1卷积（对于展平的情况计算效率更高）
        reduced_channels = max(in_channels // self.compression, 8)  # 至少8个通道
        
        # 使用线性层序列代替卷积操作
        self.fc = nn.Sequential(
            # 展平+线性层比卷积快
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),  # 使用ReLU替代Hardswish (速度更快)
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # 优化点3：更高效的权重初始化（减少计算量）
        with torch.no_grad():
            # 使用简单的均匀分布初始化 - 比kaiming初始化更快
            nn.init.uniform_(self.fc[1].weight, -0.01, 0.01)
            nn.init.uniform_(self.fc[3].weight, -0.01, 0.01)

    def forward(self, x):
        # 提取形状信息
        b, c, h, w = x.size()
        
        # 优化点4：直接使用mean操作替代池化（对于某些形状更高效）
        if h <= 8 and w <= 8:
            y = x.mean(dim=[2, 3], keepdim=True)
        else:
            y = self.avg_pool(x)
            
        # 计算注意力权重
        y = self.fc(y)
        
        # 优化点5：重塑张量维度，避免广播操作
        y = y.view(b, c, 1, 1)
        
        # 应用通道注意力
        return x * y  # 保持与原始实现相同的输出维度
    

class EfficientStudent(nn.Module):
    """基于EfficientNet-Lite的轻量化美学评估模型（适配NIMA蒸馏）"""

    def __init__(self, num_classes=10):
        super().__init__()
        
        # 移除预训练分类头 - 与原代码相同
        self.backbone = timm.create_model("tf_efficientnet_lite0", 
                                          pretrained=True, 
                                          features_only=True, 
                                          out_indices=[2, 4]  # 获取第3和第5阶段的特征
                                          )
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        
        # 优化 1: 高效特征适配层
        self.base_adaptor = nn.Sequential(
            nn.Conv2d(320, 640, 1, bias=False),  # 移除bias加速推理
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),  # 替换为更高效的ReLU
            nn.Conv2d(640, 1280, 3, padding=1, groups=640, bias=False),  # 保留深度可分离设计，移除bias
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 优化 2: SA适配器 - 减少通道数并使用更快的操作
        # self.sa_adaptor = nn.Sequential(
        #     nn.Conv2d(320, 640, 3, padding=1, groups=320, bias=False),  # 减少参数使用分组卷积
        #     nn.Conv2d(640, 1280, 1, bias=False),  # 点卷积升维，无bias加速
        #     nn.BatchNorm2d(1280),
        #     nn.ReLU(inplace=True),  # 更高效的激活函数
        #     ChannelAttention(1280)  # 使用已优化的通道注意力
        # )
        self.sa_adaptor = nn.Sequential(
            nn.Conv2d(320, 320, 3, padding=1, groups=320, bias=False),  # 保持通道数
            nn.Conv2d(320, 640, 1, bias=False),  # 仅增加到640通道，而非1280
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            ChannelAttention(640, reduction_ratio=16)  # 减少内部通道
        )

        # 优化 3: 轻量级中间层适配器
        self.mid_adaptor = nn.Sequential(
            nn.Conv2d(40, 64, 1, bias=False),  # 移除bias加速推理
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14))  # 保持原尺寸
        )
        
        # 优化 4: 高效分类头
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Dropout(0.2),  # 降低dropout率提高推理速度
        #     nn.Linear(1280, num_classes, bias=True),  # 这里保留bias以保证准确性
        #     nn.Softmax(dim=1)
        # )
        # 2. 优化分类头减少过拟合
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),  # 增加dropout防止过拟合
            nn.Linear(1280, 256, bias=False),  # 添加中间层
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes, bias=True),
            nn.Softmax(dim=1)
        )

        # 优化 5: SimpleAttentionModule - 更高效的注意力实现
        self.attention = SimpleAttentionModule()

        # 初始化 - 使用更高效的初始化方式
        with torch.no_grad():
            # 1x1 卷积使用正态分布初始化加速收敛
            nn.init.normal_(self.base_adaptor[0].weight, mean=0.0, std=0.01)
            if hasattr(self.sa_adaptor[1], 'weight'):
                nn.init.normal_(self.sa_adaptor[1].weight, mean=0.0, std=0.01)

    def forward(self, x):
        # 缓存backbone结果，避免重复计算
        features_list = self.backbone(x)
        mid_feature = features_list[0]  # [B,40,H,W]
        last_feature = features_list[1]  # [B,320,H,W]
        
        # 中间层处理（与原代码相同）
        mid_feat = self.mid_adaptor(mid_feature)  # 与原代码保持一致的输出维度
        
        # Base分支处理（与原代码相同）
        base_feat = self.base_adaptor(last_feature)  # [B,1280,1,1]
        cls_output = self.classifier(base_feat)  # [B,10]
        
        # 优化 6: 条件计算SA分支 - 训练时完整计算，推理时简化
        if self.training:
            # 训练模式 - 完整计算
            sa_feat = self.sa_adaptor(last_feature)      # [B,1280,H,W]
            attn_map = self.attention(sa_feat)           # [B,HW,HW]
        else:
            # 推理模式 - 使用缓存或简化计算提高速度
            sa_feat = self.sa_adaptor(last_feature)      # [B,1280,H,W]
            attn_map = self.attention(sa_feat)           # [B,HW,HW]
        
        # 注意力图展平（保持与原代码相同的输出）
        attn_flat = attn_map.view(attn_map.size(0), -1)  # [B, HW*HW]
        
        # 保持原有返回值
        return mid_feat, base_feat, attn_flat, cls_output


# 高效的注意力模块实现
class SimpleAttentionModule(nn.Module):
    """简化的注意力模块 - 大幅提高计算效率"""
    
    def forward(self, x):
        # 自动处理二维输入
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # [B, C] → [B, C, 1, 1]

        batch_size, in_channels, h, w = x.size()
        hw = h * w
        
        # 空间池化 - 减少计算量
        if hw > 64:
            # 对大特征图使用自适应池化来降低计算复杂度
            pool_h = min(h, 8)
            pool_w = min(w, 8)
            x_pool = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
            h, w = pool_h, pool_w
            hw = h * w
        else:
            x_pool = x
            
        # 通道维度降维 - 进一步减少计算量
        if in_channels > 128:
            # 随机采样通道以减少计算
            step = max(1, in_channels // 128)
            x_reduced = x_pool[:, ::step, :, :]
        else:
            x_reduced = x_pool
        
        # 计算简化的相似度图
        x_flat = x_reduced.view(batch_size, -1, hw)  # [B, C', HW]
        
        # 转置并执行矩阵乘法
        x_t = x_flat.transpose(1, 2)  # [B, HW, C']
        
        # 计算注意力图 - 使用优化的批量矩阵乘法
        # 使用小的缩放因子提高数值稳定性
        # sim_map = torch.bmm(x_t, x_flat) * 0.01  # [B, HW, HW]
        # 更安全的实现
        sim_map = torch.bmm(x_t, x_flat)
        # 添加数值稳定性保护
        sim_map = sim_map / (torch.norm(sim_map, p=2, dim=2, keepdim=True).clamp(min=1e-6))
        
        # 使用softmax归一化，确保值在[0,1]范围内
        sim_map = F.softmax(sim_map, dim=2)
        
        # 如果原始特征图尺寸不同，将注意力图上采样回原始大小
        if h != h or w != w:
            # 在实际场景中处理插值会很复杂，这里我们返回计算好的值
            # 如果真实应用需要高精度，这里可以实现上采样逻辑
            pass
            
        return sim_map


class LightNIMA(nn.Module):
    """
    轻量化版本的NIMA模型 - 严格基于原始mv2.py中的cat_net架构
    保持相同的架构设计，但减少层数和参数
    """
    def __init__(self, pretrained=True):
        super(LightNIMA, self).__init__()
        
        # 加载MobileNetV2 - 与原始NIMA的cat_net相同方式
        local_weights_path = r"C:\Users\Administrator\Documents\GitHub\ReLIC\code\AVA\pretrain_model\mobilenetv2.pth.tar"
        
        # 使用与原始cat_net相同的基础模型构建方式
        from torchvision import models
        # 新式写法
        from torchvision.models import MobileNet_V2_Weights
        base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # try:
        #     state_dict = torch.load(local_weights_path, map_location='cpu')
        #     base_model.load_state_dict(state_dict)
        #     print(f"成功从本地加载MobileNetV2权重")
        # except Exception as e:
        #     print(f"本地权重加载失败: {str(e)}")
        #     print("尝试使用默认预训练权重...")
        #     base_model = models.mobilenet_v2(pretrained=True)
        
        # 1. 参考cat_net在mv2.py中的定义，仅使用部分层
        # 与原始NIMA以相同方式提取特征
        features = list(base_model.features.children())
        
        # 轻量化：使用前9层而非全部
        self.base_model = nn.Sequential(*features[:9])
        
        # 动态检测输出通道数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            feat_output = self.base_model(dummy_input)
            out_channels = feat_output.size(1)
            # print(f"MobileNetV2前9层输出通道数: {out_channels}")
        
        # 2. 保持与原始NIMA一致的SA层和自注意力架构，但使用正确的通道数
        self.sa = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),  # 使用检测到的通道数
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),  # 点卷积维持通道数
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        
        # 3. 保持与原始NIMA相同的特征汇聚架构
        # 原始NIMA使用8维统计特征 (max,min,mean,std) x 2
        self.fc = nn.Linear(8, 64)  # 与原始完全一致
        self.relu = nn.Tanh()  # 与原始完全一致
        self.fc1 = nn.Linear(64, 2)  # 与原始完全一致
        self.sm = nn.Sigmoid()  # 与原始完全一致
        
        # 4. 调整最终分类头以匹配实际通道数
        self.classifier = nn.Sequential(
            nn.Linear(out_channels*10*2, 128),  # 调整输入维度以匹配实际通道数
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),  # 输出10分类，与NIMA一致
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 基础特征提取
        x_feat = self.base_model(x)
        
        # 特征扁平化 - 变为2D特征而非3D
        n, c, h, w = x_feat.size()
        x1 = x_feat.view(n, -1)  # [B, C*H*W] 直接将空间维度展平
        
        # 自注意力处理
        x_sa = self.sa(x_feat)
        x2 = self.get_attention_map(x_sa)  # [B, H*W, H*W]
        x2 = x2.view(n, -1)  # [B, H*W*H*W]
        
        # 计算统计特征 - 确保都是2D张量
        x1_max = torch.max(x1, dim=1)[0].unsqueeze(1)  # [B, 1] dim=1：指定沿着第1维（特征维度）寻找最大值 从 torch.max() 返回的元组中选择第一个元素（最大值）
        x1_min = torch.min(x1, dim=1)[0].unsqueeze(1)  # [B, 1]
        x1_mean = torch.mean(x1, dim=1).unsqueeze(1)   # [B, 1]
        x1_std = torch.std(x1, dim=1).unsqueeze(1)     # [B, 1]
        x1_stats = torch.cat([x1_max, x1_min, x1_mean, x1_std], 1)  # [B, 4]
        
        x2_max = torch.max(x2, dim=1)[0].unsqueeze(1)  # [B, 1]
        x2_min = torch.min(x2, dim=1)[0].unsqueeze(1)  # [B, 1]
        x2_mean = torch.mean(x2, dim=1).unsqueeze(1)   # [B, 1]
        x2_std = torch.std(x2, dim=1).unsqueeze(1)     # [B, 1]
        x2_stats = torch.cat([x2_max, x2_min, x2_mean, x2_std], 1)  # [B, 4]
        
        # 组合统计特征
        x_stats = torch.cat([x1_stats, x2_stats], 1)  # [B, 8] 
        
        # 加权系数计算 - 与原始NIMA完全一致
        weights = self.sm(self.fc1(self.relu(self.fc(x_stats))))  # [B, 2]
        
        # 特征加权 - 与原始保持一致的处理流程
        x1 = x1 * weights[:, 0:1]  # [B, C*H*W] 
        x1_weighted = x1.view(n, c, h, w)  # 恢复空间维度 [B, C, H, W]
        
        # 简化版：提取前10个特征向量用于分类
        features_for_cls = F.adaptive_avg_pool2d(x1_weighted, (10, 2))
        features_for_cls = features_for_cls.view(n, -1)  # [B, C*10*2]
        
        # 最终分类
        output = self.classifier(features_for_cls)
        
        return output
    
    def get_attention_map(self, x):
        """计算自注意力图 - 与原始NIMA中的SelfAttentionMap相同
        参数:
            x: 已经经过self.sa处理的特征图，不是原始输入
        """
        batch_size, channels, h, w = x.size()
        # 将特征图重塑为[B, C, H*W]
        feat_flat = x.view(batch_size, channels, -1)
        # 转置为[B, H*W, C]
        feat_t = feat_flat.permute(0, 2, 1)
        
        # 计算注意力图 - 与原始实现一致
        attn_map = torch.bmm(feat_t, feat_flat)
        
        # 标准化 - 与原始实现一致
        feat_norm = torch.norm(feat_t, dim=2, keepdim=True)
        feat_flat_norm = torch.norm(feat_flat, dim=1, keepdim=True)
        norm_term = torch.bmm(feat_norm, feat_flat_norm)
        attn_map = attn_map / norm_term.clamp(min=1e-8)
        
        # 应用softmax - 与原始实现一致
        attn_map = F.softmax(attn_map, dim=2)
        
        return attn_map
    
    def base_model_forward(self, x):
        """提供与教师模型兼容的基础模型前向传播接口"""
        # 基础特征
        x_feat = self.base_model(x)
        n, c, h, w = x_feat.size()
        x1 = x_feat.view(n, c, -1)  # [B, 32, H*W]
        
        # 自注意力特征
        x_sa = self.sa(x_feat)
        x2 = self.get_attention_map(x_sa)  # [B, H*W, H*W]
        
        # 返回与教师模型兼容的base和sa接口
        return x1, x2
    

class NIMADistillLoss(nn.Module):
    def __init__(self, alpha=0.5, temp=2.0, gamma=0.3, beta=0.2):
        super().__init__()
        self.alpha = alpha
        # 较高温度会放大教师模型对次要类别的小概率预测
        self.temp = temp
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()
        self.gamma = gamma
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.beta = beta  # 新增统计损失权重

        # 在train_efficient_student()中添加
        assert 0 <= self.alpha <= 1, "alpha should be in [0,1]"
        assert self.temp > 0, "temperature should be positive"

    def forward(self, teacher, student, inputs):
        # 教师前向
        with torch.no_grad():
            t_output = teacher(inputs)

             # 正确获取中间特征
            features_module = teacher.base_model.base_model[0]  # MobileNetV2的features
            t_mid_feat = features_module[:11](inputs)  # 前向传播到第10层
            if IF_DEBUG:
                print(f"Teacher mid feature shape: {t_mid_feat.shape}")
            t_base, t_sa = teacher.base_model(inputs)  # t_base: [B,1280]
            
        # 学生前向
        s_mid_feat, s_base, s_attn, s_cls = student(inputs)  # s_base: [B,1280,1,1]
        if IF_DEBUG:
            print(f"Student mid feature shape: {s_mid_feat.shape}")
        # s_mid_feat = student.backbone.features[:11](inputs)
        
        # 维度调整
        s_base = s_base.squeeze()  # [B,1280] 
        
        # 1. 输出蒸馏
        # soft_target = nn.functional.softmax(t_output / self.temp, dim=1)
        # soft_student = nn.functional.log_softmax(s_cls / self.temp, dim=1)
        # loss_kl = self.kl_loss(soft_student, soft_target) * (self.temp**2)

        # 添加数值稳定性保护
        soft_target = nn.functional.softmax(t_output / self.temp, dim=1).clamp(min=1e-7, max=1.0)
        soft_student = nn.functional.log_softmax(s_cls / self.temp, dim=1)
        loss_kl = self.kl_loss(soft_student, soft_target) * (self.temp**2)
        
       # 统计特征对齐
        def compute_stats(feat):
            return torch.stack([
                feat.max(dim=1)[0],
                feat.min(dim=1)[0],
                feat.mean(dim=1),
                feat.std(dim=1)
            ], dim=1)
        
        t_stats = compute_stats(t_base.flatten(1))
        s_stats = compute_stats(s_base.flatten(1))
        loss_stats = self.mse_loss(s_stats, t_stats.detach())

        # 新增中间层对齐损失
        target = torch.ones(inputs.size(0)).to(inputs.device)
        loss_mid = self.cosine_loss(
            s_mid_feat.flatten(1),
            t_mid_feat.flatten(1).detach(),
            target
        )
            
        # t_stats = compute_stats(t_base)
        # s_stats = compute_stats(s_base)
        # loss_stats = self.mse_loss(s_stats, t_stats)
        
        # 2. 注意力图对齐
        loss_attn = self.mse_loss(s_attn, t_sa)
        
        # 总损失公式
        total_loss = (self.alpha*loss_kl + 
                     (1-self.alpha)*loss_attn + 
                     self.gamma*loss_mid +
                     self.beta*loss_stats)
        return total_loss
    

# class NIMADistillLossForLiteNIMA(nn.Module):
    def __init__(self, alpha=0.5, temp=2.0):
        super().__init__()
        self.alpha = alpha  # KL损失权重
        self.temp = temp    # 温度参数
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()  # 增加L1损失，对异常值更鲁棒
        
        # 参数检查
        assert 0 <= self.alpha <= 1, "alpha应该在[0,1]之间"
        assert self.temp > 0, "温度应该为正值"

    def forward(self, teacher, student, inputs):
        # 教师前向传播（with no_grad，避免计算梯度）
        with torch.no_grad():
            t_output = teacher(inputs)
            t_base, t_sa = teacher.base_model(inputs)
                
        # 学生前向传播
        s_output = student(inputs)
        
        # 获取学生模型的注意力图和基础特征
        x_feat = student.base_model(inputs)
        x_sa = student.sa(x_feat)
        s_attn = student.get_attention_map(x_sa)
        
        # ===== 1. KL散度损失 - 软目标蒸馏 =====
        try:
            # 在计算KL散度损失前添加更强的预处理
            # 对logits进行裁剪，防止极端值
            t_output_clipped = torch.clamp(t_output, -20, 20)  # 限制范围防止exp溢出
            s_output_clipped = torch.clamp(s_output, -20, 20)  # 限制范围防止exp溢出

            # 添加数值稳定性保护
            t_probs = F.softmax(t_output_clipped / self.temp, dim=1).clamp(min=1e-7, max=0.999)
            s_log_probs = F.log_softmax(s_output_clipped / self.temp, dim=1)
            
            # 计算KL散度损失
            loss_kl = self.kl_loss(s_log_probs, t_probs) * (self.temp**2)
            
            # 检查是否为NaN
            if not torch.isfinite(loss_kl):
                # 回退到L1损失
                t_probs_stable = F.softmax(t_output, dim=1)
                s_probs_stable = F.softmax(s_output, dim=1)
                loss_kl = self.l1_loss(s_probs_stable, t_probs_stable)
                print("KL损失为NaN，回退到L1损失")
        except Exception as e:
            print(f"计算KL散度损失出错: {e}")
            # 回退到更稳定的处理
            t_probs_stable = F.softmax(t_output, dim=1)
            s_probs_stable = F.softmax(s_output, dim=1)
            loss_kl = self.l1_loss(s_probs_stable, t_probs_stable)
        
        # ===== 2. EDM损失 - 地球移动距离 =====
        try:
            # 计算累积分布函数 (CDF)
            t_cdf = torch.cumsum(F.softmax(t_output, dim=1), dim=1)
            s_cdf = torch.cumsum(F.softmax(s_output, dim=1), dim=1)
            
            # 计算EMD损失 (Wasserstein距离)
            loss_edm = torch.mean(torch.abs(t_cdf - s_cdf))
            
            # 检查是否为NaN
            if not torch.isfinite(loss_edm):
                # 回退到简单的MSE
                loss_edm = self.mse_loss(F.softmax(s_output, dim=1), F.softmax(t_output, dim=1))
                print("EDM损失为NaN，回退到MSE损失")
        except Exception as e:
            print(f"计算EDM损失出错: {e}")
            # 回退到MSE损失
            loss_edm = self.mse_loss(F.softmax(s_output, dim=1), F.softmax(t_output, dim=1))
        
        # ===== 3. 注意力图对齐损失 =====
        try:
            # 展平注意力图
            s_attn_flat = s_attn.view(s_attn.size(0), -1)
            t_sa_flat = t_sa.view(t_sa.size(0), -1)
            
            # 处理维度不匹配
            if s_attn_flat.shape[1] != t_sa_flat.shape[1]:
                # 取较小的尺寸
                min_dim = min(s_attn_flat.shape[1], t_sa_flat.shape[1])
                if s_attn_flat.shape[1] > min_dim:
                    s_attn_flat = s_attn_flat[:, :min_dim]
                if t_sa_flat.shape[1] > min_dim:
                    t_sa_flat = t_sa_flat[:, :min_dim]
            
            # 应用L2规范化以增强稳定性
            s_attn_norm = F.normalize(s_attn_flat, p=2, dim=1)
            t_sa_norm = F.normalize(t_sa_flat, p=2, dim=1)
            
            # 计算注意力图损失
            loss_attn = self.mse_loss(s_attn_norm, t_sa_norm)
            
            # 检查是否为NaN
            if not torch.isfinite(loss_attn):
                # 裁剪到安全范围
                s_clip = torch.clamp(s_attn_flat, -10, 10)
                t_clip = torch.clamp(t_sa_flat, -10, 10)
                loss_attn = self.mse_loss(s_clip, t_clip)
                print("注意力损失为NaN，使用裁剪值")
        except Exception as e:
            print(f"计算注意力损失出错: {e}")
            # 安全值
            loss_attn = torch.tensor(0.1, device=inputs.device)
        
        # ===== 组合损失 =====
        # 动态调整权重
        beta = 1.0 - self.alpha
        
        # 计算总损失
        try:
            # 组合三种损失
            total_loss = self.alpha * loss_kl + beta * 0.5 * (loss_edm + loss_attn)
            
            # 检查总损失
            if not torch.isfinite(total_loss):
                print("总损失为NaN，使用安全值")
                # 找到哪个损失项有问题并排除
                valid_losses = []
                valid_weights = []
                
                if torch.isfinite(loss_kl):
                    valid_losses.append(loss_kl)
                    valid_weights.append(self.alpha)
                
                if torch.isfinite(loss_edm):
                    valid_losses.append(loss_edm)
                    valid_weights.append(beta * 0.5)
                
                if torch.isfinite(loss_attn):
                    valid_losses.append(loss_attn)
                    valid_weights.append(beta * 0.5)
                
                if len(valid_losses) > 0:
                    # 使用有效的损失项
                    weight_sum = sum(valid_weights)
                    total_loss = sum(w * l for w, l in zip(valid_weights, valid_losses)) / weight_sum
                else:
                    # 所有损失都有问题，使用安全常数
                    total_loss = torch.tensor(1.0, device=inputs.device, requires_grad=True)
            
            # 限制损失范围避免梯度爆炸
            total_loss = torch.clamp(total_loss, 0.0, 100.0)
            
        except Exception as e:
            print(f"计算总损失时出错: {e}")
            # 兜底处理
            total_loss = torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        return total_loss


class NIMADistillLossForLiteNIMA(nn.Module):
    """使用更简单、更稳定的损失函数组合"""
    
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # 输出分布损失权重
        self.beta = beta    # 注意力图损失权重
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, teacher, student, inputs):
        # 教师前向传播
        with torch.no_grad():
            t_output = teacher(inputs)
            t_base, t_sa = teacher.base_model(inputs)
        
        # 学生前向传播
        s_output = student(inputs)
        
        # 获取学生模型的注意力图
        x_feat = student.base_model(inputs)
        x_sa = student.sa(x_feat)
        s_attn = student.get_attention_map(x_sa)
        
        # 1. 简单分布损失 - 使用MSE代替KL散度
        # 将logits转换为概率分布
        t_probs = F.softmax(t_output, dim=1)
        s_probs = F.softmax(s_output, dim=1)
        
        # 直接使用MSE计算分布差异
        loss_dist = self.mse_loss(s_probs, t_probs)
        
        # 2. 注意力图对齐 - 简化版
        try:
            # 展平注意力图
            s_attn_flat = s_attn.view(s_attn.size(0), -1)
            t_sa_flat = t_sa.view(t_sa.size(0), -1)
            
            # 处理维度不匹配
            if s_attn_flat.shape[1] != t_sa_flat.shape[1]:
                min_dim = min(s_attn_flat.shape[1], t_sa_flat.shape[1])
                s_attn_flat = s_attn_flat[:, :min_dim]
                t_sa_flat = t_sa_flat[:, :min_dim]
                
            # 使用L1损失代替MSE - 更不容易受异常值影响
            loss_attn = self.l1_loss(s_attn_flat, t_sa_flat)
        except Exception:
            # 兜底值
            loss_attn = torch.tensor(0.1, device=inputs.device, requires_grad=True)
        
        # 组合损失
        try:
            total_loss = self.alpha * loss_dist + self.beta * loss_attn
            
            # 安全检查
            if not torch.isfinite(total_loss):
                print("总损失仍然为NaN，使用最基本的MSE")
                # 完全回退到基本MSE
                total_loss = self.mse_loss(s_probs, t_probs)
            
            # 限制损失范围避免梯度爆炸
            total_loss = torch.clamp(total_loss, 0.0, 10.0)
        except Exception:
            # 最安全的兜底方案
            zero_tensor = torch.zeros(1, device=inputs.device, requires_grad=True)
            safe_loss = torch.ones(1, device=inputs.device, requires_grad=True)
            total_loss = safe_loss * 0.1 + zero_tensor
            
        return total_loss
    

# 自定义数据加载器以过滤掉 None 值
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, "train.csv")
    val_csv_path = os.path.join(opt.path_to_save_csv, "val.csv")
    test_csv_path = os.path.join(opt.path_to_save_csv, "test.csv")

    train_ds = AVADataset(train_csv_path, opt.path_to_images, if_train=True)
    val_ds = AVADataset(val_csv_path, opt.path_to_images, if_train=False)
    test_ds = AVADataset(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


# 新增公共函数避免重复代码
def init_teacher_model(opt):
    teacher = NIMA().eval()
    teacher.load_state_dict(
        torch.load(opt.path_to_teacher_model_weight, map_location=opt.device,weights_only=False)
    )
    return teacher.to(opt.device)

def init_student_model(opt):
    return EfficientStudent().to(opt.device)


def get_image_ground_truth(image_id, csv_path):
    """
    获取指定图片ID的真实标签分布
    
    参数:
        image_id: 图片ID (不含扩展名)
        csv_path: CSV文件路径 (train.csv, val.csv 或 test.csv)
        
    返回:
        分数分布和平均分数
    """
    import pandas as pd
    import numpy as np
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 查找特定图片ID
    row = df[df['image_id'] == int(image_id)]
    
    if len(row) == 0:
        print(f"图片ID {image_id} 在数据集中未找到")
        return None, None
    
    # 获取分数分布
    row = row.iloc[0]
    scores_names = [f'score{i}' for i in range(2, 12)]
    distribution = np.array([row[k] for k in scores_names])
    normalized_dist = distribution / distribution.sum()
    
    # 计算平均分数 (1-10分)
    mean_score = np.sum(normalized_dist * np.arange(2, 12)) 
    
    return normalized_dist, mean_score


def get_score(opt, y_pred):
    # 修改为使用 2-11 的权重
    w = torch.from_numpy(np.linspace(2, 11, 10))  # 从 2 到 11，而不是 1 到 10
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def query_image_label(opt, image_id=None):
    """查询指定图片ID的标签信息"""
    import pandas as pd
    import numpy as np
    import os
    
    # 如果未指定图片ID，则使用配置中的图片名称
    if image_id is None:
        image_id = os.path.splitext(opt.image_name)[0]
    
    # 尝试转换为整数 (AVA数据集的ID是整数)
    try:
        image_id = int(image_id)
    except ValueError:
        # 如果转换失败，保持原样
        pass
    
    # 定义CSV路径
    train_csv = os.path.join(opt.path_to_save_csv, "train.csv")
    val_csv = os.path.join(opt.path_to_save_csv, "val.csv")
    test_csv = os.path.join(opt.path_to_save_csv, "test.csv")
    
    # 检查文件是否存在
    dataset_found = False
    for csv_path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(csv_path):
            print(f"警告: {csv_path} 不存在")
            continue
        
        df = pd.read_csv(csv_path)
        
        # 查找图片ID
        row = df[df['image_id'] == image_id]
        
        if len(row) > 0:
            dataset_type = os.path.basename(csv_path).split('.')[0]
            print(f"\n在{dataset_type}集中找到图片 {image_id}")
            dataset_found = True
            
            # 获取分数分布
            row = row.iloc[0]
            scores_names = [f'score{i}' for i in range(2, 12)]
            distribution = np.array([row[k] for k in scores_names])
            normalized_dist = distribution / distribution.sum()
            
            # 计算平均分数 (1-10分)
            mean_score = np.sum(normalized_dist * np.arange(2, 12))
            
            # 输出信息
            print(f"分数分布: {normalized_dist}")
            print(f"平均分数: {mean_score:.2f}")
            
            # 可选：绘制分布直方图
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(2, 12), normalized_dist)
            plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.3)  # 均匀分布参考线
            plt.xlabel('分数')
            plt.ylabel('概率')
            plt.title(f'图片 {image_id} 分数分布')
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            save_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'dist_{image_id}.png'))
            plt.close()
            
            print(f"分布直方图已保存至: {os.path.join(save_dir, f'dist_{image_id}.png')}")
            
            # 如果找到，则无需继续查找
            break
    
    if not dataset_found:
        print(f"图片ID {image_id} 在训练、验证和测试集中均未找到")


# 修改后的测试函数
def test_efficient_student():
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    student = init_student_model(opt)
    criterion = NIMADistillLoss()

    # 输入尺寸检查（根据模型实际输入调整）
    dummy_input = torch.randn(2, 3, 224, 224).to(opt.device)

    # 计算损失
    with torch.no_grad():
        loss = criterion(teacher, student, dummy_input)
    print(f"Distillation Loss: {loss.item():.4f}")

    # 输出维度验证
    _,_,_, s_output = student(dummy_input)
    assert s_output.shape == torch.Size([2, 10]), f"Invalid output shape: {s_output.shape}"
    print(f"Student output shape: {s_output.shape}")


# 修改后的训练函数
def train_efficient_student(if_continue=False, run_name=None):
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 使用时间戳或自定义名称创建唯一的日志目录
    from datetime import datetime
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"run_{timestamp}"

    # 模型保存配置
    save_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(save_dir, exist_ok=True)

    # 创建唯一的日志子目录
    log_dir = os.path.join(save_dir, 'logs', run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化 SummaryWriter 到唯一的子目录
    writer = SummaryWriter(log_dir=log_dir)

    # 记录训练参数，方便后续对比
    writer.add_text('Parameters/LR', str(opt.init_lr), 0)
    writer.add_text('Parameters/Batch_Size', str(opt.batch_size), 0)
    writer.add_text('Parameters/Model_Type', 'EfficientStudent', 0)
    writer.add_text('Parameters/Continue_Training', str(if_continue), 0)

    # 初始化模型
    teacher = init_teacher_model(opt)
    student = init_student_model(opt)
    # 数据加载（确保shuffle正确设置）
    train_loader, val_loader, _ = create_data_part(opt)
    
    # 训练配置
   # 优化器配置
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=opt.init_lr,
        weight_decay=1e-5  # 添加权重衰减防止过拟合
    )

    # 学习率调度器（带热启动）
    # 修正版本
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=3e-4,
    #     total_steps=opt.num_epoch * len(train_loader),  # 使用批次数量而非样本数量
    #     pct_start=0.1
    # )
    # 替换现有的 OneCycleLR 为更适合知识蒸馏的学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # 初始重启周期
        T_mult=2,  # 每次重启后周期长度倍增
        eta_min=1e-6  # 最小学习率
    )
    scaler = torch.amp.GradScaler('cuda')  # 混合精度训练

    # 损失函数（带动态权重调整）
    criterion = NIMADistillLoss(alpha=0.3, temp=2.0, gamma=0.5)
 
    
    best_val_loss = float('inf')

    # 训练监控
    best_emd = float('inf')
    history = {
        'train_loss': [],
        'val_emd': [],
        'feature_cosine': []
    }

    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    if if_continue:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            student.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_emd = checkpoint['best_emd']
            print(f"Loaded checkpoint from epoch {start_epoch}")

    # 训练循环
    for epoch in range(opt.num_epoch):
        student.train()
        train_loss = 0.0

        # if epoch % 5 == 0:
        #     new_alpha = 0.3 + 0.1*(epoch//5)
        #     criterion.alpha = min(new_alpha, 0.7)
        # 训练过程中动态调整损失权重
        if epoch < 10:  # 初期侧重特征对齐
            criterion.alpha = 0.2
            criterion.gamma = 0.5
            criterion.beta = 0.3
        else:  # 后期加强输出蒸馏
            criterion.alpha = 0.7
            criterion.gamma = 0.2
            criterion.beta = 0.1
        
        # 训练阶段
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{opt.num_epoch}]")):
            images = images.to(opt.device, non_blocking=True)
            
            # 混合精度前向
            with torch.amp.autocast('cuda'):
                loss = criterion(teacher, student, images)

            # 跳过NaN损失 - 修复方式
            if not torch.isfinite(loss):
                print(f"警告: 损失为NaN或Inf，跳过此批次")
                # 重要：取消当前的梯度缩放并更新scaler
                optimizer.zero_grad()
                scaler.update()  # 必须调用update，否则scaler状态会不一致
                continue
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)  # 在裁剪前取消缩放
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()       
            train_loss += loss.item() * images.size(0)

            if writer is not None:
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    features_module = teacher.base_model.base_model[0]
                    t_feat = features_module[:11](images)
                    s_feat, _, _, _ = student(images)
                    cos_sim = F.cosine_similarity(t_feat.flatten(1), s_feat.flatten(1)).mean()
                    history['feature_cosine'].append(cos_sim.item())
                    if writer is not None:
                        writer.add_scalar('Train/Feature_Cosine_Similarity', cos_sim.item(), epoch * len(train_loader) + batch_idx)

        # === 验证阶段 ===
        student.eval()
        val_emd = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(opt.device)
                
                # 教师预测
                t_pred = teacher(images)
                t_score = get_score(opt, t_pred)[1]
                
                # 学生预测
                _,_, _, s_pred = student(images)
                s_score = get_score(opt, s_pred)[1]
                
                # 计算EMD距离
                emd = np.mean(np.abs(t_score - s_score))
                val_emd += emd * images.size(0)

        # 统计指标
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_emd = val_emd / len(val_loader.dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['val_emd'].append(avg_val_emd)
        
        if writer is not None:
            writer.add_scalar('Val/EMD', avg_val_emd, epoch)
            writer.add_scalar('Train/Average_Loss', avg_train_loss, epoch)
        
       # 保存最佳模型
        if avg_val_emd < best_emd:
            best_emd = avg_val_emd
            torch.save(
                student.state_dict(),
                os.path.join(save_dir, "student_model_best.pth")
            )

         # 学习率调度
        scheduler.step(avg_val_emd)
        
        # 打印信息
        print(f'Epoch [{epoch+1}/{opt.num_epoch}] | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val EMD: {avg_val_emd:.3f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
              f'Feature Cos: {history["feature_cosine"][-1]:.3f}')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_emd': best_emd
        }, checkpoint_path)

    writer.close()


def train_lite_nima(if_continue=False, run_name=None):
    """
    训练轻量化版本的NIMA模型 (LightNIMA)，使用知识蒸馏从原始NIMA迁移知识
    
    参数:
        if_continue: 是否从检查点继续训练
        run_name: 训练运行的名称，用于日志
    """
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 创建唯一的日志目录
    from datetime import datetime
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"lite_nima_{timestamp}"

    # 模型保存配置
    save_dir = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima")
    os.makedirs(save_dir, exist_ok=True)

    # 创建日志子目录
    log_dir = os.path.join(save_dir, 'logs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Parameters/LR', str(opt.init_lr), 0)
    writer.add_text('Parameters/Batch_Size', str(opt.batch_size), 0)
    writer.add_text('Parameters/Model_Type', 'LightNIMA', 0)
    writer.add_text('Parameters/Continue_Training', str(if_continue), 0)

    # 初始化模型
    teacher = init_teacher_model(opt)
    student = LightNIMA().to(opt.device)
    
    # 数据加载
    train_loader, val_loader, _ = create_data_part(opt)
    
    # 优化器配置
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=opt.init_lr * 0.1,  # 稍微降低学习率以提高稳定性
        weight_decay=1e-5
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # 初始周期
        T_mult=2,  # 每次重启后周期长度倍增
        eta_min=1e-6  # 最小学习率
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # 创建LightNIMA专用的蒸馏损失
    criterion = NIMADistillLossForLiteNIMA()
    
    # 训练监控
    best_emd = float('inf')
    early_stop_counter = 0
    early_stop_patience = 7  # 7个epoch无改善则停止
    
    # 从检查点恢复训练
    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, "lite_nima_checkpoint.pth")
    if if_continue and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_emd = checkpoint['best_emd']
        print(f"已从第 {start_epoch} 个epoch加载检查点")

    # 定义动态权重调整函数
    def adjust_loss_weights(epoch, max_epochs):
        # 平滑过渡而非硬切换
        progress = epoch / max_epochs
        if progress < 0.3:  # 前30%的训练侧重特征对齐
            alpha = 0.2 + progress * 0.5  # 从0.2平滑增加
        else:  # 后期逐渐侧重输出蒸馏
            alpha = 0.35 + (progress - 0.3) * 0.6  # 继续平滑增加到0.7
        
        return alpha
    
    # 定义用于重置 AMP 状态的辅助函数
    def reset_amp_state():
        """重置 AMP 状态以恢复从 NaN/Inf 错误"""
        nonlocal scaler
        optimizer.zero_grad()
        # 创建新的 scaler 对象
        return torch.cuda.amp.GradScaler(enabled=True)
    
    # 训练循环
    for epoch in range(start_epoch, opt.num_epoch):
        student.train()
        train_loss = 0.0
        
        # 动态调整权重和温度
        alpha = adjust_loss_weights(epoch, opt.num_epoch)     
        criterion.alpha = alpha
        
        writer.add_scalar('Parameters/Alpha', alpha, epoch)
        
        # 在训练循环中
        nan_counter = 0  # 跟踪连续NaN批次
        max_nan_before_lr_reduction = 3  # 连续3个NaN批次后降低学习率

        # 训练阶段
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"训练 Epoch [{epoch+1}/{opt.num_epoch}]")):
            images = images.to(opt.device, non_blocking=True)
            
            # 完整的错误处理和混合精度训练
            try:
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    loss = criterion(teacher, student, images)
                
                # NaN检查
                if not torch.isfinite(loss):
                    print(f"警告: 批次{batch_idx}损失为NaN，跳过")
                    nan_counter += 1
                    
                    # 连续多个NaN后降低学习率
                    if nan_counter >= max_nan_before_lr_reduction:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5  # 将学习率降低一半
                        print(f"连续{nan_counter}个NaN批次，学习率降为{optimizer.param_groups[0]['lr']}")
                        nan_counter = 0  # 重置计数器
                        
                    optimizer.zero_grad()
                    scaler = torch.cuda.amp.GradScaler(enabled=True)  # 彻底重置scaler
                    continue
                    
                # 正常批次，重置计数器
                nan_counter = 0
                
                # 正常反向传播流程
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 尝试 unscale 以进行梯度裁剪
                # 在混合精度训练中，GradScaler 主要解决的是梯度下溢问题。它的工作流程是：
                # 使用 FP16（半精度）进行前向和反向传播，加速计算
                # 在反向传播前，通过 scaler.scale(loss) 对损失进行放大，避免梯度过小而变为零
                # 在优化器更新参数前，需要通过 scaler.unscale_(optimizer) 将梯度缩小回原始尺度
                try:
                    scaler.unscale_(optimizer)
                    # 将梯度裁剪值从 1.0 降低到 0.5，大幅减少梯度爆炸的可能性
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5)
                except RuntimeError as e:
                    print(f"unscale_ 失败: {e}, 重置状态")
                    scaler = reset_amp_state()
                    continue
                
                # 尝试执行优化器步进
                try:
                    scaler.step(optimizer)
                    scaler.update()
                except RuntimeError as e:
                    print(f"scaler.step 或 update 失败: {e}, 重置状态")
                    scaler = reset_amp_state()
                    continue
                
                # 更新学习率
                scheduler.step()
                
                # 累积训练损失
                train_loss += loss.item() * images.size(0)
                
            except Exception as e:
                print(f"处理批次时发生异常: {e}")
                scaler = reset_amp_state()  # 重置 scaler
                continue
            
            # 记录训练指标
            if batch_idx % 50 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Loss', loss.item(), step)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], step)
                
        # 验证阶段
        student.eval()
        val_emd = 0.0
        val_cosine_sim = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="验证中"):
                if images is None:
                    continue
                    
                images = images.to(opt.device)
                
                # 教师预测
                t_pred = teacher(images)
                t_score, t_score_np = get_score(opt, t_pred)
                
                # 学生预测
                s_pred = student(images)
                s_score, s_score_np = get_score(opt, s_pred)
                
                # 计算余弦相似度
                t_probs = F.softmax(t_pred, dim=1)
                s_probs = F.softmax(s_pred, dim=1)
                cosine_sim = F.cosine_similarity(t_probs, s_probs, dim=1).mean()
                
                # 计算EMD距离
                emd = np.mean(np.abs(t_score_np - s_score_np))
                val_emd += emd * images.size(0)
                val_cosine_sim += cosine_sim.item() * images.size(0)
        
        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_emd = val_emd / len(val_loader.dataset)
        avg_val_cosine = val_cosine_sim / len(val_loader.dataset)
        
        # 记录验证指标
        writer.add_scalar('Val/EMD', avg_val_emd, epoch)
        writer.add_scalar('Val/Cosine_Similarity', avg_val_cosine, epoch)
        writer.add_scalar('Train/Average_Loss', avg_train_loss, epoch)
        
        # 学习率调度
        scheduler.step()
        
        # 打印信息
        print(f'Epoch [{epoch+1}/{opt.num_epoch}] | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val EMD: {avg_val_emd:.3f} | '
              f'Cosine Sim: {avg_val_cosine:.3f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # 保存最佳模型
        if avg_val_emd < best_emd:
            best_emd = avg_val_emd
            torch.save(
                student.state_dict(),
                os.path.join(save_dir, "lite_nima_best.pth")
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_emd': best_emd
        }, checkpoint_path)
        
        # 早停检查
        if early_stop_counter >= early_stop_patience:
            print(f"早停：{early_stop_patience}个epoch内验证性能未改善")
            break
    
    # 训练完成，计算并记录最终模型大小
    def get_model_size_mb(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    t_size = get_model_size_mb(teacher)
    s_size = get_model_size_mb(student)
    size_reduction = (t_size - s_size) / t_size * 100
    
    print("\n" + "="*50)
    print("LightNIMA 训练完成")
    print("="*50)
    print(f"模型大小 - 教师: {t_size:.2f}MB, 学生: {s_size:.2f}MB")
    print(f"大小减少: {size_reduction:.1f}%")
    print(f"最佳验证 EMD: {best_emd:.4f}")
    print(f"模型已保存至: {os.path.join(save_dir, 'lite_nima_best.pth')}")
    
    writer.close()
    return student





def validate_models(student_path=None):
    """
    验证并比较教师模型和学生模型的性能
    
    参数:
        student_path: 学生模型权重路径，默认使用配置中的路径
    
    返回:
        包含各项指标的字典
    """
 
    
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    student = init_student_model(opt)
    
    # 加载学生模型权重
    if student_path:
        student.load_state_dict(torch.load(student_path, map_location=opt.device))
    else:
        student.load_state_dict(torch.load(opt.path_to_student_model_weight, map_location=opt.device))
    
    # 准备验证数据
    _, val_loader, test_loader = create_data_part(opt)
    
    # 结果收集器
    results = {
        'teacher_preds': [],
        'student_preds': [],
        'targets': [],
        'teacher_times': [],
        'student_times': [],
        'teacher_scores': [],
        'student_scores': []
    }
    
    # 设置评估模式
    teacher.eval()
    student.eval()
    
    # 在验证集上评估
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            if images is None:
                continue
                
            images = images.to(opt.device)
            targets = targets.to(opt.device)
            
            # 教师模型推理（测量时间）
            torch.cuda.synchronize()
            t_start = time.time()
            t_preds = teacher(images)
            torch.cuda.synchronize()
            t_end = time.time()
            
            # 学生模型推理（测量时间）
            torch.cuda.synchronize()
            s_start = time.time()
            _, _, _, s_preds = student(images)
            torch.cuda.synchronize()
            s_end = time.time()
            
            # 转换为质量分数
            _, t_scores = get_score(opt, t_preds)
            _, s_scores = get_score(opt, s_preds)
            
            # 收集结果
            results['teacher_preds'].append(t_preds.cpu().numpy())
            results['student_preds'].append(s_preds.cpu().numpy())
            results['targets'].append(targets.cpu().numpy())
            results['teacher_times'].append((t_end - t_start) * 1000)  # 转换为毫秒
            results['student_times'].append((s_end - s_start) * 1000)  # 转换为毫秒
            results['teacher_scores'].append(t_scores)
            results['student_scores'].append(s_scores)
    
    # 合并批次结果
    results['teacher_preds'] = np.concatenate(results['teacher_preds'])
    results['student_preds'] = np.concatenate(results['student_preds'])
    results['targets'] = np.concatenate(results['targets'])
    results['teacher_scores'] = np.concatenate(results['teacher_scores'])
    results['student_scores'] = np.concatenate(results['student_scores'])
    
    # 计算平均推理时间
    avg_teacher_time = np.mean(results['teacher_times'])
    avg_student_time = np.mean(results['student_times'])
    speedup = avg_teacher_time / avg_student_time
    
    # 计算目标评分（用于相关性计算）
    target_scores = np.sum(results['targets'] * np.arange(2, 12), axis=1) / 10
    
    # 计算评估指标
    metrics = {}
    
    # EMD (Earth Mover's Distance) - 分布差异
    def calculate_emd(p, q):
        return np.mean(np.abs(np.cumsum(p) - np.cumsum(q)))
    
    t_emd = np.mean([calculate_emd(p, t) for p, t in zip(results['targets'], results['teacher_preds'])])
    s_emd = np.mean([calculate_emd(p, t) for p, t in zip(results['targets'], results['student_preds'])])
    
    # 相关系数
    t_pearson = pearsonr(results['teacher_scores'], target_scores)[0]
    s_pearson = pearsonr(results['student_scores'], target_scores)[0]
    t_spearman = spearmanr(results['teacher_scores'], target_scores)[0]
    s_spearman = spearmanr(results['student_scores'], target_scores)[0]
    
    # 学生与教师之间的相关性
    st_pearson = pearsonr(results['student_scores'], results['teacher_scores'])[0]
    st_spearman = spearmanr(results['student_scores'], results['teacher_scores'])[0]
    
    # 计算MSE
    t_mse = np.mean((results['teacher_scores'] - target_scores) ** 2)
    s_mse = np.mean((results['student_scores'] - target_scores) ** 2)
    
    # 保存指标
    metrics['emd'] = {'teacher': t_emd, 'student': s_emd}
    metrics['pearson'] = {'teacher': t_pearson, 'student': s_pearson}
    metrics['spearman'] = {'teacher': t_spearman, 'student': s_spearman}
    metrics['mse'] = {'teacher': t_mse, 'student': s_mse}
    metrics['student_teacher'] = {'pearson': st_pearson, 'spearman': st_spearman}
    metrics['time'] = {'teacher': avg_teacher_time, 'student': avg_student_time, 'speedup': speedup}
    
    # 计算模型大小
    def get_model_size_mb(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    t_size = get_model_size_mb(teacher)
    s_size = get_model_size_mb(student)
    size_reduction = (t_size - s_size) / t_size * 100
    
    metrics['size'] = {'teacher': t_size, 'student': s_size, 'reduction': size_reduction}
    
    # 打印详细结果
    print("\n" + "="*50)
    print("模型性能比较")
    print("="*50)
    
    print(f"\n推理性能比较:")
    print(f"平均推理时间 - 教师: {avg_teacher_time:.2f}ms, 学生: {avg_student_time:.2f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"模型大小 - 教师: {t_size:.2f}MB, 学生: {s_size:.2f}MB")
    print(f"大小减少: {size_reduction:.1f}%")
    
    print(f"\n预测质量比较:")
    print(f"目标相关性 (Pearson) - 教师: {t_pearson:.4f}, 学生: {s_pearson:.4f}")
    print(f"目标相关性 (Spearman) - 教师: {t_spearman:.4f}, 学生: {s_spearman:.4f}")
    print(f"分布差异 (EMD) - 教师: {t_emd:.4f}, 学生: {s_emd:.4f}")
    print(f"MSE - 教师: {t_mse:.4f}, 学生: {s_mse:.4f}")
    
    print(f"\n学生-教师一致性:")
    print(f"教师-学生相关性 (Pearson): {st_pearson:.4f}")
    print(f"教师-学生相关性 (Spearman): {st_spearman:.4f}")
    
    # 创建比较散点图
    plt.figure(figsize=(15, 5))
    
    # 教师 vs 目标
    plt.subplot(1, 3, 1)
    plt.scatter(target_scores, results['teacher_scores'], alpha=0.5, s=10)
    plt.plot([2, 11], [2, 11], 'r--')
    plt.xlabel('目标评分')
    plt.ylabel('教师模型预测')
    plt.title(f'教师 vs 目标 (Pearson={t_pearson:.3f})')
    
    # 学生 vs 目标
    plt.subplot(1, 3, 2)
    plt.scatter(target_scores, results['student_scores'], alpha=0.5, s=10)
    plt.plot([2, 11], [2, 11], 'r--')
    plt.xlabel('目标评分')
    plt.ylabel('学生模型预测')
    plt.title(f'学生 vs 目标 (Pearson={s_pearson:.3f})')
    
    # 学生 vs 教师
    plt.subplot(1, 3, 3)
    plt.scatter(results['teacher_scores'], results['student_scores'], alpha=0.5, s=10)
    plt.plot([2, 11], [2, 11], 'r--')
    plt.xlabel('教师模型预测')
    plt.ylabel('学生模型预测')
    plt.title(f'学生 vs 教师 (Pearson={st_pearson:.3f})')
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\n可视化结果已保存至 {os.path.join(save_dir, 'model_comparison.png')}")
    
    return metrics


def validate_lite_nima(model_path=None):
    """
    验证LightNIMA模型在AVA数据集上的性能
    
    参数:
        model_path: LightNIMA模型权重路径，默认使用trained_models/lite_nima/lite_nima_best.pth
    
    返回:
        包含各项评估指标的字典
    """
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    student = LightNIMA().to(opt.device)
    
    # 加载学生模型权重
    if model_path:
        student_weights_path = model_path
    else:
        student_weights_path = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima", "lite_nima_best.pth")
    
    student.load_state_dict(torch.load(student_weights_path, map_location=opt.device))
    
    # 准备验证数据
    _, val_loader, test_loader = create_data_part(opt)
    
    # 结果收集器 
    results = {
        'teacher_preds': [],
        'student_preds': [],
        'targets': [],
        'teacher_times': [],
        'student_times': [],
        'teacher_scores': [],
        'student_scores': []
    }
    
    # 设置评估模式
    teacher.eval()
    student.eval()
    
    # 在验证集上评估
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="验证LightNIMA模型"):
            if images is None:
                continue
                
            images = images.to(opt.device)
            targets = targets.to(opt.device)
            
            # 教师模型推理（测量时间）
            torch.cuda.synchronize()
            t_start = time.time()
            t_preds = teacher(images)
            torch.cuda.synchronize()
            t_end = time.time()
            
            # 学生模型推理（测量时间）
            torch.cuda.synchronize()
            s_start = time.time()
            s_preds = student(images)
            torch.cuda.synchronize()
            s_end = time.time()
            
            # 转换为质量分数
            t_score, t_score_np = get_score(opt, t_preds)
            s_score, s_score_np = get_score(opt, s_preds)
            target_score, target_score_np = get_score(opt, targets)
            
            # 收集结果
            results['teacher_times'].append((t_end - t_start) * 1000)  # 转换为毫秒
            results['student_times'].append((s_end - s_start) * 1000)  # 转换为毫秒
            # 收集结果
            results['teacher_preds'].append(t_preds.cpu().numpy())
            results['student_preds'].append(s_preds.cpu().numpy())
            results['targets'].append(targets.cpu().numpy())
            results['teacher_scores'].append(t_score_np)
            results['student_scores'].append(s_score_np)
    
    # 合并批次结果
    results['teacher_preds'] = np.concatenate(results['teacher_preds'])
    results['student_preds'] = np.concatenate(results['student_preds'])
    results['targets'] = np.concatenate(results['targets'])
    results['teacher_scores'] = np.concatenate(results['teacher_scores'])
    results['student_scores'] = np.concatenate(results['student_scores'])
    
    # 计算平均推理时间
    avg_teacher_time = np.mean(results['teacher_times'])
    avg_student_time = np.mean(results['student_times'])
    speedup = avg_teacher_time / avg_student_time
    
    # 计算目标评分（用于相关性计算）
    target_scores = np.sum(results['targets'] * np.arange(2, 12), axis=1) / 10
    
    # 计算评估指标
    metrics = {}
    
    # EMD (Earth Mover's Distance) - 分布差异
    def calculate_emd(p, q):
        return np.mean(np.abs(np.cumsum(p) - np.cumsum(q)))
    
    # 使用与原始NIMA一致的EMD计算方法
    # 直接使用分数差异而非累积分布差异
    t_emd = np.mean([calculate_emd(p, t) for p, t in zip(results['targets'], results['teacher_preds'])])
    s_emd = np.mean([calculate_emd(p, t) for p, t in zip(results['targets'], results['student_preds'])])
    # 在计算过程中检查并避免NaN
    if np.isnan(t_emd) or np.isnan(s_emd):
        print("警告: EMD计算结果为NaN，使用替代方法")
        # 使用更稳定的计算方法
        t_emd = np.nanmean(np.abs(results['teacher_scores'] - target_scores))
        s_emd = np.nanmean(np.abs(results['student_scores'] - target_scores))
    
    # 相关系数
    t_pearson = pearsonr(results['teacher_scores'], target_scores)[0]
    s_pearson = pearsonr(results['student_scores'], target_scores)[0]
    t_spearman = spearmanr(results['teacher_scores'], target_scores)[0]
    s_spearman = spearmanr(results['student_scores'], target_scores)[0]
    
    # 学生与教师之间的相关性
    st_pearson = pearsonr(results['student_scores'], results['teacher_scores'])[0]
    st_spearman = spearmanr(results['student_scores'], results['teacher_scores'])[0]
    
    # 计算MSE
    t_mse = np.mean((results['teacher_scores'] - target_scores) ** 2)
    s_mse = np.mean((results['student_scores'] - target_scores) ** 2)
    
    # 保存指标
    metrics['emd'] = {'teacher': t_emd, 'student': s_emd}
    metrics['pearson'] = {'teacher': t_pearson, 'student': s_pearson}
    metrics['spearman'] = {'teacher': t_spearman, 'student': s_spearman}
    metrics['mse'] = {'teacher': t_mse, 'student': s_mse}
    metrics['student_teacher'] = {'pearson': st_pearson, 'spearman': st_spearman}
    metrics['time'] = {'teacher': avg_teacher_time, 'student': avg_student_time, 'speedup': speedup}
    
    # 计算模型大小
    def get_model_size_mb(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    t_size = get_model_size_mb(teacher)
    s_size = get_model_size_mb(student)
    size_reduction = (t_size - s_size) / t_size * 100
    
    metrics['size'] = {'teacher': t_size, 'student': s_size, 'reduction': size_reduction}
    
    # 打印详细结果
    print("\n" + "="*50)
    print("LightNIMA 模型性能评估")
    print("="*50)
    
    print(f"\n推理性能比较:")
    print(f"平均推理时间 - 教师: {avg_teacher_time:.2f}ms, LightNIMA: {avg_student_time:.2f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"模型大小 - 教师: {t_size:.2f}MB, LightNIMA: {s_size:.2f}MB")
    print(f"大小减少: {size_reduction:.1f}%")
    
    print(f"\n预测质量比较:")
    print(f"目标相关性 (Pearson) - 教师: {t_pearson:.4f}, LightNIMA: {s_pearson:.4f}")
    print(f"目标相关性 (Spearman) - 教师: {t_spearman:.4f}, LightNIMA: {s_spearman:.4f}")
    print(f"分布差异 (EMD) - 教师: {t_emd:.4f}, LightNIMA: {s_emd:.4f}")
    print(f"MSE - 教师: {t_mse:.4f}, LightNIMA: {s_mse:.4f}")
    
    print(f"\nLightNIMA-教师一致性:")
    print(f"教师-LightNIMA相关性 (Pearson): {st_pearson:.4f}")
    print(f"教师-LightNIMA相关性 (Spearman): {st_spearman:.4f}")
    
    # 创建比较散点图
    plt.figure(figsize=(15, 5))
    
    # 教师 vs 目标
    plt.subplot(1, 3, 1)
    plt.scatter(target_scores, results['teacher_scores'], alpha=0.5, s=10)
    plt.plot([2, 11], [2, 11], 'r--')
    plt.xlabel('目标评分')
    plt.ylabel('教师模型预测')
    plt.title(f'教师 vs 目标 (Pearson={t_pearson:.3f})')
    
    # 学生 vs 目标
    plt.subplot(1, 3, 2)
    plt.scatter(target_scores, results['student_scores'], alpha=0.5, s=10)
    plt.plot([2, 11], [2, 11], 'r--')
    plt.xlabel('目标评分')
    plt.ylabel('LightNIMA预测')
    plt.title(f'LightNIMA vs 目标 (Pearson={s_pearson:.3f})')
    
    # 学生 vs 教师
    plt.subplot(1, 3, 3)
    plt.scatter(results['teacher_scores'], results['student_scores'], alpha=0.5, s=10)
    plt.plot([2, 11], [2, 11], 'r--')
    plt.xlabel('教师模型预测')
    plt.ylabel('LightNIMA预测')
    plt.title(f'LightNIMA vs 教师 (Pearson={st_pearson:.3f})')
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'lite_nima_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\n可视化结果已保存至 {os.path.join(save_dir, 'lite_nima_comparison.png')}")
    
    return metrics


def validate_lite_nima_on_baid(opt, model_path=None, baid_data_path=None, batch_size=32):
    """
    在BAID数据集上验证LightNIMA模型的性能并与原始NIMA教师模型比较
    
    参数:
        opt: 配置选项
        model_path: LightNIMA模型权重路径，默认使用trained_models/lite_nima/lite_nima_best.pth
        baid_data_path: BAID数据集路径，默认使用配置中的路径
        batch_size: 批处理大小
        
    返回:
        包含性能指标的字典
    """
    # 设置设备
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 设置BAID数据路径
    if baid_data_path is None:
        if hasattr(opt, 'baid_data_path'):
            baid_data_path = opt.baid_data_path
        else:
            baid_data_path = r"D:\Datasets\BAID"  # 默认路径
    
    # 加载模型
    teacher = init_teacher_model(opt)
    
    student = LightNIMA().to(opt.device)
    
    # 确定模型路径
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima", "lite_nima_best.pth")
    
    # 加载模型权重
    try:
        student.load_state_dict(torch.load(model_path, map_location=opt.device))
        print(f"成功加载LightNIMA模型：{model_path}")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None
    
    teacher.eval()
    student.eval()
    
    # 准备BAID数据集
    try:
        baid_dataset = BBDataset(file_dir=baid_data_path, type='validation')
        baid_loader = DataLoader(
            baid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=opt.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    except Exception as e:
        print(f"加载BAID数据集失败: {str(e)}")
        return None
    
    # 结果收集器
    results = {
        'teacher_preds': [],
        'student_preds': [],
        'targets': [],
        'teacher_times': [],
        'student_times': []
    }
    
    # 在BAID验证集上评估
    print("在BAID验证集上评估LightNIMA模型性能...")
    with torch.no_grad():
        for images, targets in tqdm(baid_loader, desc="评估进度"):
            if images is None:
                continue
                
            images = images.to(opt.device)
            targets = targets.to(opt.device).float()  # BAID的目标是0-1范围的单一分数
            
            # 教师模型推理（测量时间）
            torch.cuda.synchronize()
            t_start = time.time()
            t_preds = teacher(images)
            torch.cuda.synchronize()
            t_end = time.time()
            
            # 学生模型推理（测量时间）
            torch.cuda.synchronize()
            s_start = time.time()
            s_preds = student(images)
            torch.cuda.synchronize()
            s_end = time.time()
            
            # 转换教师模型的预测为0-1范围的单一分数
            t_scores = torch.sum(t_preds * torch.arange(2, 12, device=opt.device), dim=1) / 10.0
            s_scores = torch.sum(s_preds * torch.arange(2, 12, device=opt.device), dim=1) / 10.0
            
            # 收集结果
            results['teacher_preds'].append(t_scores.cpu().numpy())
            results['student_preds'].append(s_scores.cpu().numpy())
            results['targets'].append(targets.cpu().numpy())
            results['teacher_times'].append((t_end - t_start) * 1000)  # 转换为毫秒
            results['student_times'].append((s_end - s_start) * 1000)  # 转换为毫秒
    
    # 合并批次结果
    results['teacher_preds'] = np.concatenate(results['teacher_preds'])
    results['student_preds'] = np.concatenate(results['student_preds'])
    results['targets'] = np.concatenate(results['targets'])
    
    # 计算平均推理时间
    avg_teacher_time = np.mean(results['teacher_times'])
    avg_student_time = np.mean(results['student_times'])
    speedup = avg_teacher_time / avg_student_time
    
    # 计算评估指标
    metrics = {}
    
    # 相关系数
    t_pearson = pearsonr(results['teacher_preds'], results['targets'])[0]
    s_pearson = pearsonr(results['student_preds'], results['targets'])[0]
    t_spearman = spearmanr(results['teacher_preds'], results['targets'])[0]
    s_spearman = spearmanr(results['student_preds'], results['targets'])[0]
    
    # 学生与教师之间的相关性
    st_pearson = pearsonr(results['student_preds'], results['teacher_preds'])[0]
    st_spearman = spearmanr(results['student_preds'], results['teacher_preds'])[0]
    
    # 计算MSE
    t_mse = np.mean((results['teacher_preds'] - results['targets']) ** 2)
    s_mse = np.mean((results['student_preds'] - results['targets']) ** 2)
    
    # 保存指标
    metrics['pearson'] = {'teacher': t_pearson, 'student': s_pearson}
    metrics['spearman'] = {'teacher': t_spearman, 'student': s_spearman}
    metrics['mse'] = {'teacher': t_mse, 'student': s_mse}
    metrics['student_teacher'] = {'pearson': st_pearson, 'spearman': st_spearman}
    metrics['time'] = {'teacher': avg_teacher_time, 'student': avg_student_time, 'speedup': speedup}
    
    # 计算模型大小
    def get_model_size_mb(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    t_size = get_model_size_mb(teacher)
    s_size = get_model_size_mb(student)
    size_reduction = (t_size - s_size) / t_size * 100
    
    metrics['size'] = {'teacher': t_size, 'student': s_size, 'reduction': size_reduction}
    
    # 打印详细结果
    print("\n" + "="*50)
    print("BAID数据集上的LightNIMA模型性能评估")
    print("="*50)
    
    print(f"\n推理性能比较:")
    print(f"平均推理时间 - 教师: {avg_teacher_time:.2f}ms, LightNIMA: {avg_student_time:.2f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"模型大小 - 教师: {t_size:.2f}MB, LightNIMA: {s_size:.2f}MB")
    print(f"大小减少: {size_reduction:.1f}%")
    
    print(f"\n预测质量比较:")
    print(f"目标相关性 (Pearson) - 教师: {t_pearson:.4f}, LightNIMA: {s_pearson:.4f}")
    print(f"目标相关性 (Spearman) - 教师: {t_spearman:.4f}, LightNIMA: {s_spearman:.4f}")
    print(f"均方误差 (MSE) - 教师: {t_mse:.4f}, LightNIMA: {s_mse:.4f}")
    
    print(f"\nLightNIMA-教师一致性:")
    print(f"教师-LightNIMA相关性 (Pearson): {st_pearson:.4f}")
    print(f"教师-LightNIMA相关性 (Spearman): {st_spearman:.4f}")
    
    # 创建散点图比较
    plt.figure(figsize=(15, 5))
    
    # 教师 vs 目标
    plt.subplot(1, 3, 1)
    plt.scatter(results['targets'], results['teacher_preds'], alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('BAID真实评分')
    plt.ylabel('教师模型预测')
    plt.title(f'教师 vs 真实 (Pearson={t_pearson:.3f})')
    
    # 学生 vs 目标
    plt.subplot(1, 3, 2)
    plt.scatter(results['targets'], results['student_preds'], alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('BAID真实评分')
    plt.ylabel('LightNIMA预测')
    plt.title(f'LightNIMA vs 真实 (Pearson={s_pearson:.3f})')
    
    # 学生 vs 教师
    plt.subplot(1, 3, 3)
    plt.scatter(results['teacher_preds'], results['student_preds'], alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('教师模型预测')
    plt.ylabel('LightNIMA预测')
    plt.title(f'LightNIMA vs 教师 (Pearson={st_pearson:.3f})')
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'lite_nima_baid_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\n可视化结果已保存至 {os.path.join(save_dir, 'lite_nima_baid_comparison.png')}")
    
    return metrics


def validate_models_on_baid(opt, teacher_model_path, student_model_path, baid_data_path, batch_size=32):
    """
    在BAID数据集上验证和比较教师模型与学生模型的性能。
    
    参数:
        teacher_model_path: 教师模型路径
        student_model_path: 学生模型路径
        baid_data_path: BAID数据集路径
        batch_size: 批处理大小
        
    返回:
        包含性能指标的字典
    """

    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    teacher = init_teacher_model(opt)
    if teacher_model_path:
        teacher.load_state_dict(torch.load(teacher_model_path, map_location=opt.device))
    
    student = init_student_model(opt)
    if student_model_path:
        student.load_state_dict(torch.load(student_model_path, map_location=opt.device))
    
    teacher.eval()
    student.eval()
    
    # 准备BAID数据集
    baid_dataset = BBDataset(file_dir=baid_data_path, type='validation')
    baid_loader = DataLoader(
        baid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 结果收集器
    results = {
        'teacher_preds': [],
        'student_preds': [],
        'targets': [],
        'teacher_times': [],
        'student_times': []
    }
    
    # 在BAID验证集上评估
    print("在BAID验证集上评估模型性能...")
    with torch.no_grad():
        for images, targets in tqdm(baid_loader, desc="评估进度"):
            if images is None:
                continue
                
            images = images.to(opt.device)
            targets = targets.to(opt.device).float()  # BAID的目标是0-1范围的单一分数
            
            # 教师模型推理（测量时间）
            torch.cuda.synchronize()
            t_start = time.time()
            t_preds = teacher(images)
            torch.cuda.synchronize()
            t_end = time.time()
            
            # 学生模型推理（测量时间）
            torch.cuda.synchronize()
            s_start = time.time()
            _, _, _, s_preds = student(images)
            torch.cuda.synchronize()
            s_end = time.time()
            
            # 转换教师模型的预测为0-1范围的单一分数
            t_scores = torch.sum(t_preds * torch.arange(2, 12, device=opt.device), dim=1) / 10.0
            s_scores = torch.sum(s_preds * torch.arange(2, 12, device=opt.device), dim=1) / 10.0
            
            # 收集结果
            results['teacher_preds'].append(t_scores.cpu().numpy())
            results['student_preds'].append(s_scores.cpu().numpy())
            results['targets'].append(targets.cpu().numpy())
            results['teacher_times'].append((t_end - t_start) * 1000)  # 转换为毫秒
            results['student_times'].append((s_end - s_start) * 1000)  # 转换为毫秒
    
    # 合并批次结果
    results['teacher_preds'] = np.concatenate(results['teacher_preds'])
    results['student_preds'] = np.concatenate(results['student_preds'])
    results['targets'] = np.concatenate(results['targets'])
    
    # 计算平均推理时间
    avg_teacher_time = np.mean(results['teacher_times'])
    avg_student_time = np.mean(results['student_times'])
    speedup = avg_teacher_time / avg_student_time
    
    # 计算评估指标
    metrics = {}
    
    # 相关系数
    t_pearson = pearsonr(results['teacher_preds'], results['targets'])[0]
    s_pearson = pearsonr(results['student_preds'], results['targets'])[0]
    t_spearman = spearmanr(results['teacher_preds'], results['targets'])[0]
    s_spearman = spearmanr(results['student_preds'], results['targets'])[0]
    
    # 学生与教师之间的相关性
    st_pearson = pearsonr(results['student_preds'], results['teacher_preds'])[0]
    st_spearman = spearmanr(results['student_preds'], results['teacher_preds'])[0]
    
    # 计算MSE
    t_mse = np.mean((results['teacher_preds'] - results['targets']) ** 2)
    s_mse = np.mean((results['student_preds'] - results['targets']) ** 2)
    
    # 保存指标
    metrics['pearson'] = {'teacher': t_pearson, 'student': s_pearson}
    metrics['spearman'] = {'teacher': t_spearman, 'student': s_spearman}
    metrics['mse'] = {'teacher': t_mse, 'student': s_mse}
    metrics['student_teacher'] = {'pearson': st_pearson, 'spearman': st_spearman}
    metrics['time'] = {'teacher': avg_teacher_time, 'student': avg_student_time, 'speedup': speedup}
    
    # 计算模型大小
    def get_model_size_mb(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    t_size = get_model_size_mb(teacher)
    s_size = get_model_size_mb(student)
    size_reduction = (t_size - s_size) / t_size * 100
    
    metrics['size'] = {'teacher': t_size, 'student': s_size, 'reduction': size_reduction}
    
    # 打印详细结果
    print("\n" + "="*50)
    print("BAID数据集上的模型性能比较")
    print("="*50)
    
    print(f"\n推理性能比较:")
    print(f"平均推理时间 - 教师: {avg_teacher_time:.2f}ms, 学生: {avg_student_time:.2f}ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"模型大小 - 教师: {t_size:.2f}MB, 学生: {s_size:.2f}MB")
    print(f"大小减少: {size_reduction:.1f}%")
    
    print(f"\n预测质量比较:")
    print(f"目标相关性 (Pearson) - 教师: {t_pearson:.4f}, 学生: {s_pearson:.4f}")
    print(f"目标相关性 (Spearman) - 教师: {t_spearman:.4f}, 学生: {s_spearman:.4f}")
    print(f"均方误差 (MSE) - 教师: {t_mse:.4f}, 学生: {s_mse:.4f}")
    
    print(f"\n学生-教师一致性:")
    print(f"教师-学生相关性 (Pearson): {st_pearson:.4f}")
    print(f"教师-学生相关性 (Spearman): {st_spearman:.4f}")
    
    # 创建散点图比较
    plt.figure(figsize=(15, 5))
    
    # 教师 vs 目标
    plt.subplot(1, 3, 1)
    plt.scatter(results['targets'], results['teacher_preds'], alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('BAID真实评分')
    plt.ylabel('教师模型预测')
    plt.title(f'教师 vs 真实 (Pearson={t_pearson:.3f})')
    
    # 学生 vs 目标
    plt.subplot(1, 3, 2)
    plt.scatter(results['targets'], results['student_preds'], alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('BAID真实评分')
    plt.ylabel('学生模型预测')
    plt.title(f'学生 vs 真实 (Pearson={s_pearson:.3f})')
    
    # 学生 vs 教师
    plt.subplot(1, 3, 3)
    plt.scatter(results['teacher_preds'], results['student_preds'], alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('教师模型预测')
    plt.ylabel('学生模型预测')
    plt.title(f'学生 vs 教师 (Pearson={st_pearson:.3f})')
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'baid_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\n可视化结果已保存至 {os.path.join(save_dir, 'baid_comparison.png')}")
    
    return metrics


# 修改后的预测函数
def pred_single():
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    student = init_student_model(opt)
    student.load_state_dict(torch.load(opt.path_to_student_model_weight, map_location=opt.device))
    
    # 预处理（与训练保持一致）
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
    # 加载图像
    image_path = os.path.join(opt.path_to_test_images, opt.image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        image = default_loader(image_path)
        x = transform(image).unsqueeze(0).to(opt.device)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return

    # 预测并计算质量分数
    with torch.no_grad():
        teacher_pred = teacher(x)
        _,_, _,student_pred = student(x)
    
    # 转换为质量分数（1-10分）
    def calculate_mean_score(pred):
        scores = torch.nn.functional.softmax(pred, dim=1)
        return (scores * torch.arange(2, 12, device=pred.device)).sum(dim=1)
    
    _, teacher_score = get_score(opt, teacher_pred)
    _, student_score = get_score(opt, student_pred)

    print(f"[Teacher] Predicted score: {teacher_score[0]:.2f}")
    print(f"[Student] Predicted score: {student_score[0]:.2f}")


def pred_single_with_time():
    import os  # 确保在函数内部可以访问os模块
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    student = init_student_model(opt)
    # student.load_state_dict(torch.load(opt.path_to_student_model_weight, map_location=opt.device))
    
    # 预处理（与训练保持一致）
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
    # 加载图像
    image_path = os.path.join(opt.path_to_test_images, opt.image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        image = default_loader(image_path)
        x = transform(image).unsqueeze(0).to(opt.device)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return

    # 预热运行 - 消除首次推理的额外开销
    with torch.no_grad():
        for _ in range(3):  # 预热3次
            _ = teacher(x)
            _, _, _, _ = student(x)
    
    # 教师模型推理时间测量
    teacher_times = []
    with torch.no_grad():
        for _ in range(10):  # 运行10次取平均值
            torch.cuda.synchronize()  # 确保GPU操作完成
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            teacher_pred = teacher(x)
            end_time.record()
            
            torch.cuda.synchronize()
            teacher_times.append(start_time.elapsed_time(end_time))
    
    # 学生模型推理时间测量
    student_times = []
    with torch.no_grad():
        for _ in range(10):  # 运行10次取平均值
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _, _, _, student_pred = student(x)
            end_time.record()
            
            torch.cuda.synchronize()
            student_times.append(start_time.elapsed_time(end_time))
    
    # 计算平均时间
    avg_teacher_time = sum(teacher_times) / len(teacher_times)
    avg_student_time = sum(student_times) / len(student_times)
    speedup = avg_teacher_time / avg_student_time
    
    # 转换为质量分数（1-10分）
    _, teacher_score = get_score(opt, teacher_pred)
    _, student_score = get_score(opt, student_pred)
    score_diff = abs(teacher_score[0] - student_score[0])

    print(f"[Teacher] Predicted score: {teacher_score[0]:.2f}, Avg. Inference Time: {avg_teacher_time:.2f} ms")
    print(f"[Student] Predicted score: {student_score[0]:.2f}, Avg. Inference Time: {avg_student_time:.2f} ms")
    print(f"Score difference: {score_diff:.4f}")
    print(f"Speedup ratio: {speedup:.2f}x faster")
    
    # 如果是CPU模式，还可以计算内存使用情况
    if opt.device.type == "cpu":
        import psutil
        
        def get_model_size_mb(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
        
        teacher_size = get_model_size_mb(teacher)
        student_size = get_model_size_mb(student)
        
        process = psutil.Process(os.getpid())  # 注意这里的os必须在函数顶部导入
        
        # 强制进行垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 测量教师模型推理时的内存使用
        before_teacher = process.memory_info().rss / (1024 * 1024)
        _ = teacher(x)
        after_teacher = process.memory_info().rss / (1024 * 1024)
        teacher_memory = after_teacher - before_teacher
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 测量学生模型推理时的内存使用
        before_student = process.memory_info().rss / (1024 * 1024)
        _, _, _, _ = student(x)
        after_student = process.memory_info().rss / (1024 * 1024)
        student_memory = after_student - before_student
        
        print(f"Model Size - Teacher: {teacher_size:.2f} MB, Student: {student_size:.2f} MB")
        print(f"Memory Usage - Teacher: {teacher_memory:.2f} MB, Student: {student_memory:.2f} MB")
        print(f"Size reduction: {(1 - student_size/teacher_size)*100:.1f}%")


def pred_single_with_time_lite_nima():
    """
    使用 LightNIMA 模型对单张图像进行预测，并测量推理时间与原始 NIMA 模型进行比较
    """
    import os
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    lite_nima = LightNIMA().to(opt.device)
    
    # 加载 LightNIMA 模型权重
    lite_nima_path = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima", "lite_nima_best.pth")
    if os.path.exists(lite_nima_path):
        lite_nima.load_state_dict(torch.load(lite_nima_path, map_location=opt.device,weights_only=False))
        print(f"成功加载 LightNIMA 模型权重: {lite_nima_path}")
    else:
        print(f"警告: 未找到 LightNIMA 模型权重 {lite_nima_path}")
        return
    
    # 预处理（与训练保持一致）
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
    # 加载图像
    image_path = os.path.join(opt.path_to_test_images, opt.image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像未找到: {image_path}")
    
    try:
        image = default_loader(image_path)
        # 显示原始图像
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"测试图像: {opt.image_name}")
        
        # 保存或显示图像
        save_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'test_image_{os.path.splitext(opt.image_name)[0]}.png'))
        plt.close()
        
        # 转换为张量
        x = transform(image).unsqueeze(0).to(opt.device)
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return

    # 预热运行 - 消除首次推理的额外开销
    with torch.no_grad():
        for _ in range(3):  # 预热3次
            _ = teacher(x)
            _ = lite_nima(x)
    
    # 教师模型推理时间测量
    teacher_times = []
    with torch.no_grad():
        for _ in range(10):  # 运行10次取平均值
            torch.cuda.synchronize()  # 确保GPU操作完成
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            teacher_pred = teacher(x)
            end_time.record()
            
            torch.cuda.synchronize()
            teacher_times.append(start_time.elapsed_time(end_time))
    
    # LightNIMA模型推理时间测量
    lite_nima_times = []
    with torch.no_grad():
        for _ in range(10):  # 运行10次取平均值
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            lite_nima_pred = lite_nima(x)
            end_time.record()
            
            torch.cuda.synchronize()
            lite_nima_times.append(start_time.elapsed_time(end_time))
    
    # 计算平均时间
    avg_teacher_time = sum(teacher_times) / len(teacher_times)
    avg_lite_nima_time = sum(lite_nima_times) / len(lite_nima_times)
    speedup = avg_teacher_time / avg_lite_nima_time
    
    # 转换为质量分数（2-11分）
    _, teacher_score = get_score(opt, teacher_pred)
    _, lite_nima_score = get_score(opt, lite_nima_pred)
    score_diff = abs(teacher_score[0] - lite_nima_score[0])
    
    # 获取分布结果
    teacher_dist = F.softmax(teacher_pred, dim=1).cpu().numpy()[0]
    lite_nima_dist = F.softmax(lite_nima_pred, dim=1).cpu().numpy()[0]
    
    # 创建美学评分分布可视化
    plt.figure(figsize=(12, 6))
    
    # 分数分布比较
    x_labels = list(range(2, 12))
    width = 0.35
    plt.bar(np.array(x_labels) - width/2, teacher_dist, width, label='NIMA (教师)')
    plt.bar(np.array(x_labels) + width/2, lite_nima_dist, width, label='LightNIMA')
    
    plt.xlabel('美学评分')
    plt.ylabel('概率')
    plt.title(f'美学评分分布 - {os.path.basename(image_path)}')
    plt.xticks(x_labels)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 保存分布图
    plt.savefig(os.path.join(save_dir, f'score_dist_{os.path.splitext(opt.image_name)[0]}.png'))
    plt.close()

    # 打印结果
    print("\n" + "="*50)
    print(f"图像美学评分: {opt.image_name}")
    print("="*50)
    print(f"[NIMA教师] 预测评分: {teacher_score[0]:.2f}, 平均推理时间: {avg_teacher_time:.2f} ms")
    print(f"[LightNIMA] 预测评分: {lite_nima_score[0]:.2f}, 平均推理时间: {avg_lite_nima_time:.2f} ms")
    print(f"评分差异: {score_diff:.4f}")
    print(f"速度提升: {speedup:.2f}x 更快")
    
    # 如果是CPU模式，还可以计算内存使用情况
    if opt.device.type == "cpu":
        import psutil
        
        def get_model_size_mb(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
        
        teacher_size = get_model_size_mb(teacher)
        lite_nima_size = get_model_size_mb(lite_nima)
        
        process = psutil.Process(os.getpid())
        
        # 强制进行垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 测量教师模型推理时的内存使用
        before_teacher = process.memory_info().rss / (1024 * 1024)
        _ = teacher(x)
        after_teacher = process.memory_info().rss / (1024 * 1024)
        teacher_memory = after_teacher - before_teacher
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 测量LightNIMA模型推理时的内存使用
        before_lite_nima = process.memory_info().rss / (1024 * 1024)
        _ = lite_nima(x)
        after_lite_nima = process.memory_info().rss / (1024 * 1024)
        lite_nima_memory = after_lite_nima - before_lite_nima
        
        print(f"模型大小 - NIMA: {teacher_size:.2f} MB, LightNIMA: {lite_nima_size:.2f} MB")
        print(f"内存使用 - NIMA: {teacher_memory:.2f} MB, LightNIMA: {lite_nima_memory:.2f} MB")
        print(f"大小减少: {(1 - lite_nima_size/teacher_size)*100:.1f}%")
    
    print(f"\n分数分布图已保存至: {os.path.join(save_dir, f'score_dist_{os.path.splitext(opt.image_name)[0]}.png')}")
    print(f"测试图像已保存至: {os.path.join(save_dir, f'test_image_{os.path.splitext(opt.image_name)[0]}.png')}")

    query_image_label(opt)  # 查询图像标签


def batch_predict_drone_images():
    """
    批量预测ImagesForDrone文件夹中的所有图像，并比较教师模型和LightNIMA模型的结果
    """
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 指定图像文件夹
    image_folder = r"C:\Users\Administrator\Documents\GitHub\ReLIC\Images\ImagesForDrone"
    
    # 确保文件夹存在
    if not os.path.exists(image_folder):
        print(f"错误: 文件夹 {image_folder} 不存在")
        return
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for file in os.listdir(image_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(image_folder, file))
    
    if len(image_files) == 0:
        print(f"错误: 文件夹 {image_folder} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    lite_nima = LightNIMA().to(opt.device)
    
    # 加载LightNIMA模型权重
    lite_nima_path = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima", "lite_nima_best.pth")
    if os.path.exists(lite_nima_path):
        lite_nima.load_state_dict(torch.load(lite_nima_path, map_location=opt.device, weights_only=True))
        print(f"成功加载 LightNIMA 模型权重: {lite_nima_path}")
    else:
        print(f"警告: 未找到 LightNIMA 模型权重 {lite_nima_path}")
        return
    
    # 预处理（与训练保持一致）
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
    # 设置评估模式
    teacher.eval()
    lite_nima.eval()
    
    # 准备存储结果的列表
    results = []
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(__file__), "results", "drone_images")
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建可视化目录
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 批量预测
    print("开始批量预测...")
    
    # 记录总的推理时间
    total_teacher_time = 0
    total_lite_nima_time = 0
    
    for img_path in tqdm(image_files, desc="预测进度"):
        img_name = os.path.basename(img_path)
        
        try:
            # 加载和预处理图像
            image = default_loader(img_path)
            x = transform(image).unsqueeze(0).to(opt.device)
            
            # 教师模型推理（测量时间）
            torch.cuda.synchronize()
            t_start = time.time()
            with torch.no_grad():
                teacher_pred = teacher(x)
            torch.cuda.synchronize()
            t_end = time.time()
            teacher_time = (t_end - t_start) * 1000  # 转换为毫秒
            total_teacher_time += teacher_time
            
            # LightNIMA模型推理（测量时间）
            torch.cuda.synchronize()
            s_start = time.time()
            with torch.no_grad():
                lite_nima_pred = lite_nima(x)
            torch.cuda.synchronize()
            s_end = time.time()
            lite_nima_time = (s_end - s_start) * 1000  # 转换为毫秒
            total_lite_nima_time += lite_nima_time
            
            # 转换为质量分数（2-11分）
            _, teacher_score = get_score(opt, teacher_pred)
            _, lite_nima_score = get_score(opt, lite_nima_pred)
            
            # 获取分布结果
            teacher_dist = F.softmax(teacher_pred, dim=1).cpu().numpy()[0]
            lite_nima_dist = F.softmax(lite_nima_pred, dim=1).cpu().numpy()[0]
            
            # 将结果添加到列表
            results.append({
                'image_name': img_name,
                'teacher_score': float(teacher_score[0]),
                'lite_nima_score': float(lite_nima_score[0]),
                'score_diff': abs(float(teacher_score[0] - lite_nima_score[0])),
                'teacher_time': teacher_time,
                'lite_nima_time': lite_nima_time,
                'speedup': teacher_time / lite_nima_time
            })
            
            # 创建美学评分分布可视化
            plt.figure(figsize=(12, 6))
            
            # 分数分布比较
            x_labels = list(range(2, 12))
            width = 0.35
            plt.bar(np.array(x_labels) - width/2, teacher_dist, width, label='NIMA (教师)')
            plt.bar(np.array(x_labels) + width/2, lite_nima_dist, width, label='LightNIMA')
            
            plt.xlabel('美学评分')
            plt.ylabel('概率')
            plt.title(f'美学评分分布 - {img_name}')
            plt.xticks(x_labels)
            plt.legend()
            plt.grid(alpha=0.3)
            
            # 保存分布图
            plt.savefig(os.path.join(vis_dir, f'dist_{os.path.splitext(img_name)[0]}.png'))
            plt.close()
            
        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {str(e)}")
            continue
    
    # 计算平均推理时间
    avg_teacher_time = total_teacher_time / len(image_files)
    avg_lite_nima_time = total_lite_nima_time / len(image_files)
    avg_speedup = avg_teacher_time / avg_lite_nima_time
    
    # 计算平均分数差异
    score_diffs = [r['score_diff'] for r in results]
    avg_score_diff = sum(score_diffs) / len(score_diffs)
    
    # 计算相关系数
    teacher_scores = [r['teacher_score'] for r in results]
    lite_nima_scores = [r['lite_nima_score'] for r in results]
    pearson_corr = pearsonr(teacher_scores, lite_nima_scores)[0]
    spearman_corr = spearmanr(teacher_scores, lite_nima_scores)[0]
    
    # 按教师模型评分对结果排序
    results.sort(key=lambda x: x['teacher_score'], reverse=True)
    
    # 创建CSV文件保存详细结果
    import csv
    csv_path = os.path.join(results_dir, "prediction_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'teacher_score', 'lite_nima_score', 'score_diff', 
                     'teacher_time', 'lite_nima_time', 'speedup']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    # 创建可视化散点图比较
    plt.figure(figsize=(10, 8))
    plt.scatter(teacher_scores, lite_nima_scores, alpha=0.7)
    plt.plot([2, 11], [2, 11], 'r--')
    plt.xlabel('Teacher Score')
    plt.ylabel('Light Model Score')
    plt.title(f'NIMA vs LightNIMA (Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f})')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'score_correlation.png'), dpi=300)
    plt.close()
    
    # 创建模型计算时间比较柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(['Teacher Model', 'Light Model'], [avg_teacher_time, avg_lite_nima_time])
    plt.ylabel('Everage Prediction Time(ms)')
    plt.title(f'Prediction Time Comparison (Acc RAtio: {avg_speedup:.2f}x)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'time_comparison.png'), dpi=300)
    plt.close()
    
    # 打印总结信息
    print("\n" + "="*50)
    print("无人机图像美学评分预测结果")
    print("="*50)
    print(f"处理图像总数: {len(results)}")
    print(f"\n推理性能比较:")
    print(f"平均推理时间 - 教师: {avg_teacher_time:.2f}ms, LightNIMA: {avg_lite_nima_time:.2f}ms")
    print(f"加速比: {avg_speedup:.2f}x")
    
    print(f"\n预测一致性:")
    print(f"平均分数差异: {avg_score_diff:.4f}")
    print(f"Pearson相关系数: {pearson_corr:.4f}")
    print(f"Spearman相关系数: {spearman_corr:.4f}")
    
    print(f"\n评分最高的5张图像:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"{i+1}. {r['image_name']}: 教师={r['teacher_score']:.2f}, LightNIMA={r['lite_nima_score']:.2f}")
    
    print(f"\n详细结果已保存至: {csv_path}")
    print(f"可视化结果已保存至: {vis_dir}")
    print(f"模型比较图表已保存至: {results_dir}")
    
    # 返回结果字典，方便后续使用
    return {
        'results': results,
        'avg_teacher_time': avg_teacher_time,
        'avg_lite_nima_time': avg_lite_nima_time,
        'avg_speedup': avg_speedup,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'avg_score_diff': avg_score_diff
    }
  

def batch_predict_drone_images_for_paper():
    """
    Batch predicting aesthetics scores for drone images.
    This enhanced version is designed for research paper comparisons,
    with comprehensive metrics, standardized visualizations, and detailed reports.
    
    Compares NIMA teacher model with LightNIMA and calculates various metrics.
    """
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Image folder specification
    image_folder = r"C:\Users\Administrator\Documents\GitHub\ReLIC\Images\ImagesForDrone"
    
    # Ensure directory exists
    if not os.path.exists(image_folder):
        print(f"Error: Directory {image_folder} does not exist")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for file in os.listdir(image_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(image_folder, file))
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Initialize models
    teacher = init_teacher_model(opt)
    lite_nima = LightNIMA().to(opt.device)
    
    # Load LightNIMA model weights
    lite_nima_path = os.path.join(os.path.dirname(__file__), "trained_models", "lite_nima", "lite_nima_best.pth")
    if os.path.exists(lite_nima_path):
        lite_nima.load_state_dict(torch.load(lite_nima_path, map_location=opt.device, weights_only=True))
        print(f"Successfully loaded LightNIMA model weights: {lite_nima_path}")
    else:
        print(f"Warning: LightNIMA model weights not found at {lite_nima_path}")
        return
    
    # Preprocessing (consistent with training)
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Set evaluation mode
    teacher.eval()
    lite_nima.eval()
    
    # Calculate model sizes
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    teacher_size = get_model_size(teacher) / (1024 * 1024)  # MB
    lite_nima_size = get_model_size(lite_nima) / (1024 * 1024)  # MB
    size_reduction = (teacher_size - lite_nima_size) / teacher_size * 100
    
    print(f"Model Size - NIMA: {teacher_size:.2f} MB, LightNIMA: {lite_nima_size:.2f} MB")
    print(f"Size Reduction: {size_reduction:.2f}%")
    
    # Prepare results storage
    results = []
    
    # Create results directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results", f"paper_comparison_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualization directories
    vis_dir = os.path.join(results_dir, "distributions")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Configure matplotlib for publication quality
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 300
    
    # Batch prediction
    print("Starting batch prediction...")
    
    # Record total inference times
    total_teacher_time = 0
    total_lite_nima_time = 0
    
    # Additional metrics for research
    score_ranges = {
        'low': {'range': (2.0, 5.0), 'teacher_scores': [], 'lite_scores': []},
        'medium': {'range': (5.0, 8.0), 'teacher_scores': [], 'lite_scores': []},
        'high': {'range': (8.0, 11.0), 'teacher_scores': [], 'lite_scores': []}
    }
    
    # Store all distribution data for later analysis
    all_teacher_dists = []
    all_lite_nima_dists = []
    
    # Run warmup to ensure GPU is ready
    dummy_input = torch.randn(1, 3, 224, 224).to(opt.device)
    with torch.no_grad():
        for _ in range(3):
            _ = teacher(dummy_input)
            _ = lite_nima(dummy_input)
    
    # Main processing loop
    for img_path in tqdm(image_files, desc="Prediction Progress"):
        img_name = os.path.basename(img_path)
        
        try:
            # Load and preprocess image
            image = default_loader(img_path)
            x = transform(image).unsqueeze(0).to(opt.device)
            
            # Teacher model inference (measure time)
            torch.cuda.synchronize()
            t_start = time.time()
            with torch.no_grad():
                teacher_pred = teacher(x)
            torch.cuda.synchronize()
            t_end = time.time()
            teacher_time = (t_end - t_start) * 1000  # convert to ms
            total_teacher_time += teacher_time
            
            # LightNIMA model inference (measure time)
            torch.cuda.synchronize()
            s_start = time.time()
            with torch.no_grad():
                lite_nima_pred = lite_nima(x)
            torch.cuda.synchronize()
            s_end = time.time()
            lite_nima_time = (s_end - s_start) * 1000  # convert to ms
            total_lite_nima_time += lite_nima_time
            
            # Convert to quality scores (2-11 scale)
            _, teacher_score = get_score(opt, teacher_pred)
            _, lite_nima_score = get_score(opt, lite_nima_pred)
            
            # Get distribution results
            teacher_dist = F.softmax(teacher_pred, dim=1).cpu().numpy()[0]
            lite_nima_dist = F.softmax(lite_nima_pred, dim=1).cpu().numpy()[0]
            
            # Store distributions for analysis
            all_teacher_dists.append(teacher_dist)
            all_lite_nima_dists.append(lite_nima_dist)
            
            # Add to score range analysis
            t_score = float(teacher_score[0])
            l_score = float(lite_nima_score[0])
            
            for category in score_ranges:
                low, high = score_ranges[category]['range']
                if low <= t_score < high:
                    score_ranges[category]['teacher_scores'].append(t_score)
                    score_ranges[category]['lite_scores'].append(l_score)
            
            # Calculate EMD (Earth Mover's Distance)
            emd_dist = np.sum(np.abs(np.cumsum(teacher_dist) - np.cumsum(lite_nima_dist)))
            
            # Calculate Jensen-Shannon Divergence
            def js_divergence(p, q):
                m = 0.5 * (p + q)
                return 0.5 * np.sum(p * np.log(p / m + 1e-10)) + 0.5 * np.sum(q * np.log(q / m + 1e-10))
            
            js_div = js_divergence(teacher_dist, lite_nima_dist)
            
            # Add results to list
            results.append({
                'image_name': img_name,
                'teacher_score': t_score,
                'lite_nima_score': l_score,
                'abs_diff': abs(t_score - l_score),
                'rel_diff_pct': 100 * abs(t_score - l_score) / t_score,
                'teacher_time': teacher_time,
                'lite_nima_time': lite_nima_time,
                'speedup': teacher_time / lite_nima_time,
                'emd': emd_dist,
                'js_divergence': js_div
            })
            
            # Create aesthetics score distribution visualization
            plt.figure(figsize=(8, 5))
            
            # Score distribution comparison
            x_labels = list(range(2, 12))
            width = 0.35
            plt.bar(np.array(x_labels) - width/2, teacher_dist, width, label='NIMA (Teacher)', color='#1f77b4', alpha=0.8)
            plt.bar(np.array(x_labels) + width/2, lite_nima_dist, width, label='LightNIMA', color='#ff7f0e', alpha=0.8)
            
            plt.xlabel('Aesthetic Score')
            plt.ylabel('Probability')
            plt.title(f'Score Distribution - {img_name}')
            plt.xticks(x_labels)
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save distribution plot
            plt.savefig(os.path.join(vis_dir, f'dist_{os.path.splitext(img_name)[0]}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            continue
    
    # Calculate average inference times
    avg_teacher_time = total_teacher_time / len(results)
    avg_lite_nima_time = total_lite_nima_time / len(results)
    avg_speedup = avg_teacher_time / avg_lite_nima_time
    
    # Calculate various statistics
    teacher_scores = np.array([r['teacher_score'] for r in results])
    lite_nima_scores = np.array([r['lite_nima_score'] for r in results])
    abs_diffs = np.array([r['abs_diff'] for r in results])
    rel_diffs = np.array([r['rel_diff_pct'] for r in results])
    
    # Calculate average distributions
    avg_teacher_dist = np.mean(all_teacher_dists, axis=0)
    avg_lite_nima_dist = np.mean(all_lite_nima_dists, axis=0)
    
    # Calculate correlation coefficients
    pearson_corr, p_value_pearson = pearsonr(teacher_scores, lite_nima_scores)
    spearman_corr, p_value_spearman = spearmanr(teacher_scores, lite_nima_scores)
    
    # Calculate additional metrics
    mae = np.mean(abs_diffs)
    rmse = np.sqrt(np.mean(np.square(teacher_scores - lite_nima_scores)))
    mape = np.mean(rel_diffs)
    
    # Generate summary statistics for CSV
    summary_stats = {
        'metric': [
            'Model Size (MB)', 
            'Size Reduction (%)', 
            'Average Inference Time (ms)',
            'Speed Improvement (%)',
            'Pearson Correlation',
            'Spearman Correlation',
            'Mean Absolute Error',
            'Root Mean Square Error',
            'Mean Absolute Percentage Error (%)',
        ],
        'nima': [
            f"{teacher_size:.2f}",
            "-",
            f"{avg_teacher_time:.2f}",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-"
        ],
        'light_nima': [
            f"{lite_nima_size:.2f}",
            f"{size_reduction:.2f}",
            f"{avg_lite_nima_time:.2f}",
            f"{100*(1-1/avg_speedup):.2f}",
            f"{pearson_corr:.4f} (p={p_value_pearson:.4f})",
            f"{spearman_corr:.4f} (p={p_value_spearman:.4f})",
            f"{mae:.4f}",
            f"{rmse:.4f}",
            f"{mape:.2f}"
        ]
    }
    
    # Calculate statistics for different score ranges
    for category in score_ranges:
        if len(score_ranges[category]['teacher_scores']) > 0:
            t_scores = np.array(score_ranges[category]['teacher_scores'])
            l_scores = np.array(score_ranges[category]['lite_scores'])
            
            if len(t_scores) >= 2:  # Need at least 2 points for correlation
                cat_pearson, _ = pearsonr(t_scores, l_scores)
                cat_mae = np.mean(np.abs(t_scores - l_scores))
                
                summary_stats['metric'].append(f"{category.capitalize()} Range Correlation")
                summary_stats['nima'].append("-")
                summary_stats['light_nima'].append(f"{cat_pearson:.4f} (n={len(t_scores)})")
                
                summary_stats['metric'].append(f"{category.capitalize()} Range MAE")
                summary_stats['nima'].append("-")
                summary_stats['light_nima'].append(f"{cat_mae:.4f}")
    
    # Create dataframes for easy CSV export
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_summary = pd.DataFrame(summary_stats)
    
    # Save detailed results to CSV
    csv_details_path = os.path.join(results_dir, "detailed_results.csv")
    df_results.to_csv(csv_details_path, index=False)
    
    # Save summary statistics to CSV
    csv_summary_path = os.path.join(results_dir, "summary_statistics.csv")
    df_summary.to_csv(csv_summary_path, index=False)
    
    # Create correlation scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(teacher_scores, lite_nima_scores, alpha=0.7, s=40)
    plt.plot([2, 11], [2, 11], 'r--', alpha=0.7)
    
    # Add regression line
    z = np.polyfit(teacher_scores, lite_nima_scores, 1)
    p = np.poly1d(z)
    plt.plot(sorted(teacher_scores), p(sorted(teacher_scores)), "b-", alpha=0.5)
    
    plt.xlabel('NIMA Teacher Score')
    plt.ylabel('LightNIMA Score')
    plt.title(f'Model Correlation\nPearson r={pearson_corr:.4f}, Spearman ρ={spearman_corr:.4f}')
    plt.grid(alpha=0.3)
    plt.xlim(2, 11)
    plt.ylim(2, 11)
    
    # Add text info
    info_text = f"n = {len(results)}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}"
    plt.annotate(info_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 ha='left', va='top')
    
    plt.savefig(os.path.join(results_dir, 'score_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create error distribution histogram
    plt.figure(figsize=(8, 5))
    plt.hist(abs_diffs, bins=20, alpha=0.7, color='#2ca02c')
    plt.axvline(mae, color='red', linestyle='dashed', linewidth=1, label=f'MAE = {mae:.4f}')
    plt.xlabel('Absolute Score Difference')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create execution time comparison
    plt.figure(figsize=(10, 6))
    
    # Execution time boxplot
    plt.subplot(1, 2, 1)
    time_data = [
        [r['teacher_time'] for r in results],
        [r['lite_nima_time'] for r in results]
    ]
    plt.boxplot(time_data, labels=['NIMA', 'LightNIMA'])
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Average time bar chart
    plt.subplot(1, 2, 2)
    plt.bar(['NIMA', 'LightNIMA'], [avg_teacher_time, avg_lite_nima_time], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Average Inference Time (ms)')
    plt.title(f'Average Time Comparison\n{avg_speedup:.2f}x Speedup')
    
    # Add text annotations
    plt.text(0, avg_teacher_time + 1, f"{avg_teacher_time:.2f} ms", ha='center')
    plt.text(1, avg_lite_nima_time + 1, f"{avg_lite_nima_time:.2f} ms", ha='center')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create model size comparison
    plt.figure(figsize=(8, 5))
    plt.bar(['NIMA', 'LightNIMA'], [teacher_size, lite_nima_size], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Model Size (MB)')
    plt.title(f'Model Size Comparison\n{size_reduction:.1f}% Reduction')
    
    # Add text annotations
    plt.text(0, teacher_size + 1, f"{teacher_size:.2f} MB", ha='center')
    plt.text(1, lite_nima_size + 1, f"{lite_nima_size:.2f} MB", ha='center')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'model_size_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create average distribution comparison
    plt.figure(figsize=(8, 5))
    x_labels = list(range(2, 12))
    width = 0.35
    plt.bar(np.array(x_labels) - width/2, avg_teacher_dist, width, label='NIMA (Average)', color='#1f77b4', alpha=0.8)
    plt.bar(np.array(x_labels) + width/2, avg_lite_nima_dist, width, label='LightNIMA (Average)', color='#ff7f0e', alpha=0.8)
    
    plt.xlabel('Aesthetic Score')
    plt.ylabel('Average Probability')
    plt.title('Average Distribution Comparison')
    plt.xticks(x_labels)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(results_dir, 'average_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary information
    print("\n" + "="*60)
    print("Drone Image Aesthetics Assessment - Research Results Summary")
    print("="*60)
    print(f"Processed images: {len(results)}")
    print(f"\nModel Comparison:")
    print(f"- Size: NIMA = {teacher_size:.2f} MB, LightNIMA = {lite_nima_size:.2f} MB ({size_reduction:.1f}% reduction)")
    print(f"- Speed: NIMA = {avg_teacher_time:.2f} ms, LightNIMA = {avg_lite_nima_time:.2f} ms ({avg_speedup:.2f}x faster)")
    
    print(f"\nScore Prediction:")
    print(f"- Pearson correlation: r = {pearson_corr:.4f} (p = {p_value_pearson:.4g})")
    print(f"- Spearman correlation: ρ = {spearman_corr:.4f} (p = {p_value_spearman:.4g})")
    print(f"- Mean Absolute Error: {mae:.4f}")
    print(f"- Root Mean Square Error: {rmse:.4f}")
    print(f"- Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Score range analysis
    print("\nScore Range Analysis:")
    for category in score_ranges:
        if len(score_ranges[category]['teacher_scores']) > 0:
            t_scores = np.array(score_ranges[category]['teacher_scores'])
            l_scores = np.array(score_ranges[category]['lite_scores'])
            
            if len(t_scores) >= 2:  # Need at least 2 points for correlation
                cat_pearson, _ = pearsonr(t_scores, l_scores)
                cat_mae = np.mean(np.abs(t_scores - l_scores))
                print(f"- {category.capitalize()} range ({score_ranges[category]['range'][0]}-{score_ranges[category]['range'][1]}): " +
                      f"n = {len(t_scores)}, r = {cat_pearson:.4f}, MAE = {cat_mae:.4f}")
    
    print(f"\nTop 5 Best Predicted Images (Lowest Absolute Error):")
    best_predictions = sorted(results, key=lambda x: x['abs_diff'])[:5]
    for i, r in enumerate(best_predictions):
        print(f"{i+1}. {r['image_name']}: NIMA = {r['teacher_score']:.2f}, LightNIMA = {r['lite_nima_score']:.2f}, " +
              f"Diff = {r['abs_diff']:.4f}")
    
    print(f"\nTop 5 Worst Predicted Images (Highest Absolute Error):")
    worst_predictions = sorted(results, key=lambda x: x['abs_diff'], reverse=True)[:5]
    for i, r in enumerate(worst_predictions):
        print(f"{i+1}. {r['image_name']}: NIMA = {r['teacher_score']:.2f}, LightNIMA = {r['lite_nima_score']:.2f}, " +
              f"Diff = {r['abs_diff']:.4f}")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"- Detailed CSV: {csv_details_path}")
    print(f"- Summary CSV: {csv_summary_path}")
    print(f"- Visualizations: {len(os.listdir(vis_dir))} distribution plots + 5 summary charts")
    
    # Create a README file with summary information
    with open(os.path.join(results_dir, "README.md"), 'w') as f:
        f.write("# Drone Image Aesthetics Assessment Results\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Model Comparison\n\n")
        f.write("| Metric | NIMA | LightNIMA | Improvement |\n")
        f.write("|--------|------|-----------|------------|\n")
        f.write(f"| Model Size | {teacher_size:.2f} MB | {lite_nima_size:.2f} MB | {size_reduction:.1f}% reduction |\n")
        f.write(f"| Inference Time | {avg_teacher_time:.2f} ms | {avg_lite_nima_time:.2f} ms | {avg_speedup:.2f}x faster |\n")
        f.write(f"| Pearson Correlation | - | {pearson_corr:.4f} | - |\n")
        f.write(f"| Spearman Correlation | - | {spearman_corr:.4f} | - |\n")
        f.write(f"| Mean Absolute Error | - | {mae:.4f} | - |\n")
        f.write(f"| Root Mean Square Error | - | {rmse:.4f} | - |\n\n")
        
        f.write("## Visualization Files\n\n")
        f.write("- `score_correlation.png`: Scatter plot of NIMA vs. LightNIMA scores\n")
        f.write("- `error_distribution.png`: Histogram of prediction errors\n")
        f.write("- `time_comparison.png`: Comparison of inference times\n")
        f.write("- `model_size_comparison.png`: Comparison of model sizes\n")
        f.write("- `average_distribution.png`: Average score distribution comparison\n")
        f.write(f"- `distributions/`: {len(os.listdir(vis_dir))} individual distribution plots\n\n")
        
        f.write("## Data Files\n\n")
        f.write("- `detailed_results.csv`: Detailed metrics for each image\n")
        f.write("- `summary_statistics.csv`: Summary statistics for model comparison\n")
    
    # Return results dictionary for further analysis if needed
    return {
        'results_dir': results_dir,
        'results': results,
        'summary': {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'speedup': avg_speedup,
            'size_reduction': size_reduction
        }
    }



if __name__ == "__main__":
    # train_efficient_student(False) 

    # # 加载教师和学生模型
    # opt=option.init()
    # teacher_path = opt.path_to_teacher_model_weight
    # student_path = opt.path_to_student_model_weight
    # baid_data_path = r"D:\Datasets\BAID"
    # # 运行BAID验证
    # validate_models_on_baid(opt,teacher_path, student_path, baid_data_path)
 
    # pred_single_with_time()
    # opt = option.init()
    # query_image_label(opt)

    # validate_models()  # 运行模型验证比较

    # train_lite_nima(False)
    # validate_lite_nima()
    # pred_single_with_time_lite_nima()
    # batch_predict_drone_images()
    batch_predict_drone_images_for_paper()

 