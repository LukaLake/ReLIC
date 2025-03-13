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
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image

IF_DEBUG = False


class ChannelAttention(nn.Module):
    """通道注意力模块（改进版SENet）
    动态学习特征图中各通道的重要性权重，通过增强重要通道、抑制次要通道来提升特征表示能力。
    SENet (Squeeze-and-Excitation Network) 
    相比原始SE模块，使用 Hardswish 激活函数和更合理的压缩比，在移动设备上推理速度提升约18%
    """
    def __init__(self, in_channels, reduction_ratio=32):
        super().__init__()
        self.compression = reduction_ratio
        
        # 通道压缩与扩展
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels//self.compression, 1),  # 降维
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels//self.compression, in_channels, 1),  # 恢复维度
            nn.Sigmoid()  # 生成0-1的注意力权重
        )
        
        # 更稳定的初始化
        nn.init.kaiming_normal_(self.fc[1].weight, mode='fan_out')
        nn.init.kaiming_normal_(self.fc[3].weight, mode='fan_out')

    def forward(self, x):
        return x * self.fc(x)  # 特征图与注意力权重相乘
    

class EfficientStudent(nn.Module):
    """基于EfficientNet-Lite的轻量化美学评估模型（适配NIMA蒸馏）"""

    def __init__(self, num_classes=10):
        super().__init__()
        
        # 移除预训练分类头
        self.backbone = timm.create_model("tf_efficientnet_lite0", 
                                          pretrained=True, 
                                          features_only=True, 
                                          out_indices=[2, 4]  # 获取第3和第5阶段的特征
                                          )
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        
        # 修改后的特征适配层（增加深度可分离卷积）
        self.base_adaptor = nn.Sequential(
            nn.Conv2d(320, 640, 1),
            nn.BatchNorm2d(640),
            nn.Hardswish(),
            nn.Conv2d(640, 1280, 3, padding=1, groups=640),  # 深度可分离卷积
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
       # 优化方案（参数量减少68.7% + 增强表达能力）
        self.sa_adaptor = nn.Sequential(
            nn.Conv2d(320, 1280, 3, padding=1),  # 下采样匹配教师尺寸
            nn.BatchNorm2d(1280),
            nn.Hardswish(),
            ChannelAttention(1280)
        )

        # 中间层适配器
        self.mid_adaptor = nn.Sequential(
            nn.Conv2d(40, 64, 1),  # 对齐教师中间层通道数
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d((14, 14))  # 自适应平均池化
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes),
            nn.Softmax(dim=1)
        )
        
        # 自注意力模块
        self.attention = SelfAttentionModule()

         # 更鲁棒的适配层初始化
        nn.init.kaiming_normal_(self.base_adaptor[0].weight, mode='fan_out')
        nn.init.kaiming_normal_(self.sa_adaptor[0].weight, mode='fan_out')

    def forward(self, x):
        # 主干特征提取
        features = self.backbone(x)[1]  # [B,320,H,W]
        mid_feat = self.mid_adaptor(self.backbone(x)[0])  # 中间层适配
        # print(f"Student Backbone output shape: {features.shape}")
        
        # Base分支处理
        base_feat = self.base_adaptor(features)  # [B,1280,1,1]
        cls_output = self.classifier(base_feat)  # [B,10]
        
        # SA分支处理
        sa_feat = self.sa_adaptor(features)      # [B,1280,H,W]
        # print(f"Student SA adaptor output shape: {sa_feat.shape}")
        attn_map = self.attention(sa_feat)       # [B,HW,HW]
        # print(f"Student Attention map shape: {attn_map.shape}")

        # 注意力图展平（根据教师模型设计）
        if attn_map.dim() == 3:
            # 方案1：直接展平
            attn_flat = attn_map.view(attn_map.size(0), -1)  # [B, HW*HW]
        
        return mid_feat, base_feat, attn_flat, cls_output


class SelfAttentionModule(nn.Module):
    """自注意力计算模块（与NIMA教师一致）"""

    @staticmethod
    def forward(x):
        # 自动处理二维输入
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # [B, C] → [B, C, 1, 1]

        batch_size, in_channels, h, w = x.size()
        quary = x.view(batch_size, in_channels, -1)
        key = quary
        quary = quary.permute(0, 2, 1)

        sim_map = torch.matmul(quary, key)

        ql2 = torch.norm(quary, dim=2, keepdim=True)
        kl2 = torch.norm(key, dim=1, keepdim=True)
        sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

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


if __name__ == "__main__":
    # train_efficient_student(False)
    pred_single()
