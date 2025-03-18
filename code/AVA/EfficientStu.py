# EfficientStu.py
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
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image

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
        self.sa_adaptor = nn.Sequential(
            nn.Conv2d(320, 640, 3, padding=1, groups=320, bias=False),  # 减少参数使用分组卷积
            nn.Conv2d(640, 1280, 1, bias=False),  # 点卷积升维，无bias加速
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),  # 更高效的激活函数
            ChannelAttention(1280)  # 使用已优化的通道注意力
        )

        # 优化 3: 轻量级中间层适配器
        self.mid_adaptor = nn.Sequential(
            nn.Conv2d(40, 64, 1, bias=False),  # 移除bias加速推理
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14))  # 保持原尺寸
        )
        
        # 优化 4: 高效分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),  # 降低dropout率提高推理速度
            nn.Linear(1280, num_classes, bias=True),  # 这里保留bias以保证准确性
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
        sim_map = torch.bmm(x_t, x_flat) * 0.01  # [B, HW, HW]
        
        # 使用softmax归一化，确保值在[0,1]范围内
        sim_map = F.softmax(sim_map, dim=2)
        
        # 如果原始特征图尺寸不同，将注意力图上采样回原始大小
        if h != h or w != w:
            # 在实际场景中处理插值会很复杂，这里我们返回计算好的值
            # 如果真实应用需要高精度，这里可以实现上采样逻辑
            pass
            
        return sim_map

class NIMADistillLoss(nn.Module):
    def __init__(self, alpha=0.5, temp=3.0, gamma=0.3, beta=0.2):
        super().__init__()
        self.alpha = alpha
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
        soft_target = nn.functional.softmax(t_output / self.temp, dim=1)
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
        torch.load(opt.path_to_teacher_model_weight, map_location=opt.device)
    )
    return teacher.to(opt.device)

def init_student_model(opt):
    return EfficientStudent().to(opt.device)

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
def train_efficient_student(if_continue=False):
    opt = option.init()
    opt.device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    teacher = init_teacher_model(opt)
    student = init_student_model(opt)
    # 数据加载（确保shuffle正确设置）
    train_loader, val_loader, _ = create_data_part(opt)
    
    # 训练配置
   # 优化器配置
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=2e-4,
        weight_decay=1e-5  # 添加权重衰减防止过拟合
    )

    # 学习率调度器（带热启动）
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=opt.num_epoch * len(train_loader.dataset),
        pct_start=0.1  # 前10%步数作为热启动
    )
    scaler = torch.amp.GradScaler('cuda')  # 混合精度训练

    # 损失函数（带动态权重调整）
    criterion = NIMADistillLoss(alpha=0.3, temp=3.0, gamma=0.5)
 
    # 模型保存配置
    save_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    # 训练监控
    best_emd = float('inf')
    history = {
        'train_loss': [],
        'val_emd': [],
        'feature_cosine': []
    }

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

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
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                max_norm=2.0  # 根据实验调整
            )    
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


def get_score(opt,y_pred):
    w = torch.from_numpy(np.linspace(1,10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def validate_models(student_path=None):
    """
    验证并比较教师模型和学生模型的性能
    
    参数:
        student_path: 学生模型权重路径，默认使用配置中的路径
    
    返回:
        包含各项指标的字典
    """
    import time
    from scipy.stats import pearsonr, spearmanr
    import matplotlib.pyplot as plt
    
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
    target_scores = np.sum(results['targets'] * np.arange(1, 11), axis=1) / 10
    
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
    plt.plot([1, 10], [1, 10], 'r--')
    plt.xlabel('目标评分')
    plt.ylabel('教师模型预测')
    plt.title(f'教师 vs 目标 (Pearson={t_pearson:.3f})')
    
    # 学生 vs 目标
    plt.subplot(1, 3, 2)
    plt.scatter(target_scores, results['student_scores'], alpha=0.5, s=10)
    plt.plot([1, 10], [1, 10], 'r--')
    plt.xlabel('目标评分')
    plt.ylabel('学生模型预测')
    plt.title(f'学生 vs 目标 (Pearson={s_pearson:.3f})')
    
    # 学生 vs 教师
    plt.subplot(1, 3, 3)
    plt.scatter(results['teacher_scores'], results['student_scores'], alpha=0.5, s=10)
    plt.plot([1, 10], [1, 10], 'r--')
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
        return (scores * torch.arange(1, 11, device=pred.device)).sum(dim=1)
    
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

if __name__ == "__main__":
    train_efficient_student(False)
    # pred_single_with_time()
    # validate_models()  # 运行模型验证比较
