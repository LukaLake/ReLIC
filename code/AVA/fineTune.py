import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.optim import RAdam

from models.relic2_model import NIMA, NIMA_Finetune
from dataset import AVADataset, BBDataset
current_path = os.path.abspath(__file__)
directory_name = os.path.dirname(current_path)
import sys
sys.path.append("..")
from util import EDMLoss,AverageMeter
import option as option
import shutil

def load_pretrained_nima(finetune_model, pretrained_path):
    """
    加载预训练的NIMA模型权重到NIMA_Finetune模型
    
    参数:
        finetune_model: NIMA_Finetune的实例
        pretrained_path: 预训练NIMA模型的路径
    
    返回:
        加载了预训练权重的NIMA_Finetune模型
    """
    # 加载预训练的NIMA模型权重
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    # 获取NIMA_Finetune模型的状态字典
    model_dict = finetune_model.state_dict()
    
    # 过滤预训练权重（只保留名称匹配的参数）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    # 更新模型权重
    model_dict.update(pretrained_dict)
    finetune_model.load_state_dict(model_dict)
    
    return finetune_model


# 将collate_fn移至全局作用域
def collate_fn(batch):
    # 过滤掉None值
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        # 如果批次中没有有效样本，返回一个空批次
        return None
    # 否则使用默认的collate函数
    return torch.utils.data.dataloader.default_collate(batch)


def print_model_layers(model):
    print("模型层结构:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")




def finetune_on_baid(pretrained_path, baid_data_path, save_dir, num_epochs=20, batch_size=32, lr = 1e-4 , if_continue=False):
    """
    在BAID数据集上微调NIMA模型
    
    参数:
        pretrained_path: 预训练NIMA模型路径
        baid_data_path: BAID数据集路径
        save_dir: 保存模型的目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        if_continue: 是否从保存的检查点继续训练
    """
    # 设备配置
    opt = option.init()
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化微调模型并加载预训练权重
    finetune_model = NIMA_Finetune()
    finetune_model = load_pretrained_nima(finetune_model, pretrained_path)
    finetune_model = finetune_model.to(device)
    
    # 在训练开始前打印
    print_model_layers(finetune_model)

    # 加载BAID数据集
    train_dataset = BBDataset(file_dir=baid_data_path, type='train')
    val_dataset = BBDataset(file_dir=baid_data_path, type='validation')

    # 在创建DataLoader时使用：
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 训练配置
    # 第一阶段：只微调baid_head
    for name, param in finetune_model.named_parameters():
        if 'baid_head' not in name:
            param.requires_grad = False
    
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, finetune_model.parameters()),
    #     lr=lr,
    #     weight_decay=1e-4
    # )

    optimizer = RAdam(
        filter(lambda p: p.requires_grad, finetune_model.parameters()),
        lr=lr,
        weight_decay=1e-3
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # 每次减半学习率
        patience=5,  # 5个epoch没有提升就降低学习率
        min_lr=1e-6
    )
    criterion = nn.MSELoss()
    
    # TensorBoard初始化
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    # 训练状态
    best_val_loss = float('inf')
    best_epoch = -1  # 添加记录最佳epoch的变量
    start_epoch = 0
    
    # 如果继续训练，加载检查点
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    if if_continue and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        finetune_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"继续训练：从第 {start_epoch} 轮开始")
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        finetune_model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"训练轮次 [{epoch+1}/{num_epochs}]")):
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            
            # 前向传播
            outputs = finetune_model(inputs, mode='baid')
            loss = criterion(outputs.squeeze(), targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 记录批次级别的训练指标
            if batch_idx % 20 == 0:
                writer.add_scalar('Train/Batch_Loss', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # 验证阶段
        finetune_model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="验证中"):
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                
                # 前向传播
                outputs = finetune_model(inputs, mode='baid')
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # 收集预测和目标值，用于计算相关系数
                val_preds.append(outputs.squeeze().cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # 计算皮尔逊相关系数和斯皮尔曼相关系数
        val_preds_flat = np.concatenate(val_preds)
        val_targets_flat = np.concatenate(val_targets)
        pearson_corr = pearsonr(val_preds_flat, val_targets_flat)[0]
        spearman_corr = spearmanr(val_preds_flat, val_targets_flat)[0]
        
        # 记录每轮的指标
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation/Pearson_Correlation', pearson_corr, epoch)
        writer.add_scalar('Validation/Spearman_Correlation', spearman_corr, epoch)
        
        # 打印训练状态
        print(f"轮次 [{epoch+1}/{num_epochs}] - "
              f"训练损失: {avg_train_loss:.4f}, "
              f"验证损失: {avg_val_loss:.4f}, "
              f"Pearson: {pearson_corr:.4f}, "
              f"Spearman: {spearman_corr:.4f}")
        
        # 保存检查点，用于恢复训练
        torch.save({
            'epoch': epoch,
            'model_state_dict': finetune_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'val_pearson': pearson_corr,
            'val_spearman': spearman_corr
        }, checkpoint_path)
        
        # 如果当前模型是最佳模型，保存它
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch  # 记录最佳epoch
            torch.save(finetune_model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        
        # 更新学习率
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
    
        # 第二阶段：10轮后，解冻更多层
        if epoch == 10:
            print("第二阶段：解冻FC层和基础模型的最后几层")
            for name, param in finetune_model.named_parameters():
                # 解冻所有fc层、baid_head及基础模型的最后一些层
                if any(x in name for x in ['fc', 'baid_head', 'base_model.model.8', 'base_model.model.9']):
                    param.requires_grad = True
            
            # 更新优化器以包含解冻的参数
            optimizer = RAdam(
                filter(lambda p: p.requires_grad, finetune_model.parameters()),
                lr=lr * 0.5,  # 降低学习率
                weight_decay=1e-3
            )
            
            # 重置学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )

        # 第三阶段：20轮后，解冻更深的层
        if epoch == 20:
            print("第三阶段：解冻更多基础模型层")
            for name, param in finetune_model.named_parameters():
                # 解冻基础模型的更多层，但保留前几层冻结
                if 'base_model.model.0' not in name and 'base_model.model.1' not in name:
                    param.requires_grad = True
                    
            # 再次更新优化器，使用更小的学习率
            optimizer = RAdam(
                filter(lambda p: p.requires_grad, finetune_model.parameters()),
                lr=lr * 0.1,  # 进一步降低学习率
                weight_decay=1e-4  # 微调时可以减小权重衰减
            )
            
            # 重置学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
    
    # 训练结束
    writer.close()
    print(f"微调完成。最佳验证损失: {best_val_loss:.4f}，出现在第 {best_epoch+1}/{num_epochs} 轮")

    # 也保存最后一个模型
    torch.save(finetune_model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    
    # 返回微调后的模型
    return finetune_model


def pred_single_finetuned(model_path, image_path, original_model_path=None, device=None):
    """
    使用微调后的模型预测单个图片的美学评分，并与原始模型进行对比
    
    参数:
        model_path: 微调后的模型权重文件路径
        image_path: 要预测的图像路径
        original_model_path: 原始AVA模型权重文件路径
        device: 计算设备，如果为None则自动选择
    
    返回:
        预测的美学评分 (0-1范围)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载微调后的模型
    finetune_model = NIMA_Finetune()
    finetune_model.load_state_dict(torch.load(model_path, map_location=device))
    finetune_model.eval()
    finetune_model = finetune_model.to(device)
    
    # 加载原始NIMA模型（如果提供了路径）
    original_model = None
    if original_model_path and os.path.exists(original_model_path):
        original_model = NIMA()
        original_model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model.eval()
        original_model = original_model.to(device)
    
    # 预处理图像
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    try:
        # 加载并转换图像
        image = default_loader(image_path)
        x = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None
    
    # 预测
    with torch.no_grad():
        # 微调模型的BAID预测
        baid_output = finetune_model(x, mode='baid').item()
        
        # 微调模型的AVA预测
        finetune_ava_output = finetune_model(x, mode='ava')
        finetune_ava_score = (finetune_ava_output * torch.arange(1, 11, device=device)).sum(dim=1).item()
        finetune_ava_normalized = finetune_ava_score / 10.0
        
        # 原始模型的AVA预测（如果可用）
        original_ava_score = None
        original_ava_normalized = None
        if original_model:
            original_ava_output = original_model(x)
            original_ava_score = (original_ava_output * torch.arange(1, 11, device=device)).sum(dim=1).item()
            original_ava_normalized = original_ava_score / 10.0
        
    # 打印结果
    print(f"============= 预测结果 =============")
    print(f"图像: {os.path.basename(image_path)}")
    print(f"BAID 模型预测分数 (0-1): {baid_output:.4f} ({baid_output*10:.2f}/10)")
    print(f"微调模型的AVA预测 (0-1): {finetune_ava_normalized:.4f} ({finetune_ava_score:.2f}/10)")
    
    if original_ava_score is not None:
        print(f"原始AVA模型预测 (0-1): {original_ava_normalized:.4f} ({original_ava_score:.2f}/10)")
    
    # 显示图像与评分（如果环境支持）
    try:
        from matplotlib import pyplot as plt
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"美学评分: {baid_output*10:.2f}/10")
        plt.show()
    except:
        pass
    
    return baid_output

def batch_predict(model_path, image_dir, output_csv=None, threshold=0.6):
    """
    批量预测目录中所有图像的美学评分，并可以根据阈值筛选出高质量图像
    
    参数:
        model_path: 模型权重文件路径
        image_dir: 包含图像的目录
        output_csv: 可选，保存结果的CSV文件路径
        threshold: 高质量图像的阈值 (0-1)
    
    返回:
        预测结果的列表，每个元素是包含图像名称和分数的字典
    """
    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
        return []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = NIMA_Finetune()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    
    # 预处理图像
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    if not image_files:
        print(f"目录中没有找到支持的图像文件: {image_dir}")
        return []
    
    # 批量预测
    results = []
    for image_file in tqdm(image_files, desc="预测进度"):
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # 加载并转换图像
            image = default_loader(image_path)
            x = transform(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                # 使用BAID模式获取评分
                score = model(x, mode='baid').item()
                
            # 将结果添加到列表中
            results.append({
                'image_name': image_file,
                'score': score,
                'high_quality': score >= threshold
            })
        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {str(e)}")
    
    # 统计高质量图像
    high_quality_images = [r for r in results if r['high_quality']]
    print(f"总共处理图像: {len(results)}")
    print(f"高质量图像 (分数 >= {threshold}): {len(high_quality_images)}")
    
    # 保存结果到CSV（如果指定）
    if output_csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"结果已保存到: {output_csv}")
    
    return results


def validate_models(original_model_path, finetuned_model_path, 
                   ava_val_csv=None, baid_data_dir=None, 
                   batch_size=32, num_workers=8, device=None):
    """
    对比原始NIMA模型和微调后的模型在AVA和BAID数据集上的表现
    
    参数:
        original_model_path: 原始NIMA模型路径
        finetuned_model_path: 微调后的NIMA_Finetune模型路径
        ava_val_csv: AVA验证集CSV文件路径
        baid_data_dir: BAID验证集数据目录
        batch_size: 批处理大小
        num_workers: 数据加载的工作线程数
        device: 计算设备
    
    返回:
        包含各项评估指标的字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # 1. 加载模型
    print("加载模型...")
    # 原始NIMA模型
    original_model = NIMA()
    original_model.load_state_dict(torch.load(original_model_path, map_location=device))
    original_model.eval()
    original_model = original_model.to(device)
    
    # 微调后的模型
    finetuned_model = NIMA_Finetune()
    finetuned_model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    finetuned_model.eval()
    finetuned_model = finetuned_model.to(device)
    
    results = {}
    
    # 2. 如果提供了AVA验证集路径，在AVA验证集上评估
    if ava_val_csv and os.path.exists(ava_val_csv):
        print("\n在AVA验证集上评估...")
        # 加载AVA验证集
        opt = option.init()
        ava_val_dataset = AVADataset(ava_val_csv, opt.path_to_images, if_train=False)
        ava_val_loader = DataLoader(
            ava_val_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 评估指标
        criterion = EDMLoss()
        criterion.to(device)
        
        # 收集预测和目标值
        original_pred_scores = []
        finetuned_pred_scores = []
        finetuned_baid_scores = []
        true_scores = []
        
        with torch.no_grad():
            for x, y in tqdm(ava_val_loader, desc="AVA验证集评估"):
                if x is None:
                    continue
                
                x = x.to(device)
                y = y.to(device).float()
                
                # 原始模型预测
                original_output = original_model(x)
                original_score = (original_output * torch.arange(1, 11, device=device)).sum(dim=1)
                original_pred_scores.extend(original_score.cpu().numpy())
                
                # 微调模型预测 - AVA模式
                finetuned_ava_output = finetuned_model(x, mode='ava')
                finetuned_ava_score = (finetuned_ava_output * torch.arange(1, 11, device=device)).sum(dim=1)
                finetuned_pred_scores.extend(finetuned_ava_score.cpu().numpy())
                
                # 微调模型预测 - BAID模式
                finetuned_baid_output = finetuned_model(x, mode='baid')
                finetuned_baid_scores.extend((finetuned_baid_output.squeeze() * 10).cpu().numpy())  # 转换为1-10分
                
                # 真实分数
                true_score = (y * torch.arange(1, 11, device=device)).sum(dim=1)
                true_scores.extend(true_score.cpu().numpy())
        
        # 计算评估指标
        # PLCC - 皮尔逊线性相关系数
        original_plcc = pearsonr(original_pred_scores, true_scores)[0]
        finetuned_ava_plcc = pearsonr(finetuned_pred_scores, true_scores)[0]
        finetuned_baid_plcc = pearsonr(finetuned_baid_scores, true_scores)[0]
        
        # SRCC - 斯皮尔曼等级相关系数
        original_srcc = spearmanr(original_pred_scores, true_scores)[0]
        finetuned_ava_srcc = spearmanr(finetuned_pred_scores, true_scores)[0]
        finetuned_baid_srcc = spearmanr(finetuned_baid_scores, true_scores)[0]
        
        # MSE - 均方误差
        original_mse = np.mean((np.array(original_pred_scores) - np.array(true_scores)) ** 2)
        finetuned_ava_mse = np.mean((np.array(finetuned_pred_scores) - np.array(true_scores)) ** 2)
        finetuned_baid_mse = np.mean((np.array(finetuned_baid_scores) - np.array(true_scores)) ** 2)
        
        # 记录结果
        results['AVA'] = {
            'original': {
                'PLCC': original_plcc,
                'SRCC': original_srcc,
                'MSE': original_mse
            },
            'finetuned_ava': {
                'PLCC': finetuned_ava_plcc,
                'SRCC': finetuned_ava_srcc,
                'MSE': finetuned_ava_mse
            },
            'finetuned_baid': {
                'PLCC': finetuned_baid_plcc,
                'SRCC': finetuned_baid_srcc,
                'MSE': finetuned_baid_mse
            }
        }
        
        # 打印结果
        print("\nAVA验证集评估结果:")
        print(f"原始模型 - PLCC: {original_plcc:.4f}, SRCC: {original_srcc:.4f}, MSE: {original_mse:.4f}")
        print(f"微调模型(AVA模式) - PLCC: {finetuned_ava_plcc:.4f}, SRCC: {finetuned_ava_srcc:.4f}, MSE: {finetuned_ava_mse:.4f}")
        print(f"微调模型(BAID模式) - PLCC: {finetuned_baid_plcc:.4f}, SRCC: {finetuned_baid_srcc:.4f}, MSE: {finetuned_baid_mse:.4f}")
    
    # 3. 如果提供了BAID验证集路径，在BAID验证集上评估
    if baid_data_dir and os.path.exists(baid_data_dir):
        print("\n在BAID验证集上评估...")
        # 加载BAID验证集 - 注意这里baid_val_dir应指向BAID数据集的根目录，而不是具体的验证集目录
        baid_val_dataset = BBDataset(file_dir=baid_data_dir, type='validation')
        baid_val_loader = DataLoader(
            baid_val_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 收集预测和目标值
        original_pred_scores = []
        finetuned_ava_scores = []
        finetuned_baid_scores = []
        true_scores = []
        
        with torch.no_grad():
            for x, y in tqdm(baid_val_loader, desc="BAID验证集评估"):
                if x is None:
                    continue
                    
                x = x.to(device)
                y = y.to(device).float()  # y是单一分数(0-1范围)
                
                # 原始模型预测
                original_output = original_model(x)
                original_score = (original_output * torch.arange(1, 11, device=device)).sum(dim=1) / 10.0  # 归一化到0-1
                original_pred_scores.extend(original_score.cpu().numpy())
                
                # 微调模型预测 - AVA模式
                finetuned_ava_output = finetuned_model(x, mode='ava')
                finetuned_ava_score = (finetuned_ava_output * torch.arange(1, 11, device=device)).sum(dim=1) / 10.0  # 归一化到0-1
                finetuned_ava_scores.extend(finetuned_ava_score.cpu().numpy())
                
                # 微调模型预测 - BAID模式
                finetuned_baid_output = finetuned_model(x, mode='baid')
                finetuned_baid_scores.extend(finetuned_baid_output.squeeze().cpu().numpy())
                
                # 真实分数
                true_scores.extend(y.cpu().numpy())
        
        # 计算评估指标
        # PLCC - 皮尔逊线性相关系数
        original_plcc = pearsonr(original_pred_scores, true_scores)[0]
        finetuned_ava_plcc = pearsonr(finetuned_ava_scores, true_scores)[0]
        finetuned_baid_plcc = pearsonr(finetuned_baid_scores, true_scores)[0]
        
        # SRCC - 斯皮尔曼等级相关系数
        original_srcc = spearmanr(original_pred_scores, true_scores)[0]
        finetuned_ava_srcc = spearmanr(finetuned_ava_scores, true_scores)[0]
        finetuned_baid_srcc = spearmanr(finetuned_baid_scores, true_scores)[0]
        
        # MSE - 均方误差
        original_mse = np.mean((np.array(original_pred_scores) - np.array(true_scores)) ** 2)
        finetuned_ava_mse = np.mean((np.array(finetuned_ava_scores) - np.array(true_scores)) ** 2)
        finetuned_baid_mse = np.mean((np.array(finetuned_baid_scores) - np.array(true_scores)) ** 2)
        
        # 记录结果
        results['BAID'] = {
            'original': {
                'PLCC': original_plcc,
                'SRCC': original_srcc,
                'MSE': original_mse
            },
            'finetuned_ava': {
                'PLCC': finetuned_ava_plcc,
                'SRCC': finetuned_ava_srcc,
                'MSE': finetuned_ava_mse
            },
            'finetuned_baid': {
                'PLCC': finetuned_baid_plcc,
                'SRCC': finetuned_baid_srcc,
                'MSE': finetuned_baid_mse
            }
        }
        
        # 打印结果
        print("\nBAID验证集评估结果:")
        print(f"原始模型 - PLCC: {original_plcc:.4f}, SRCC: {original_srcc:.4f}, MSE: {original_mse:.4f}")
        print(f"微调模型(AVA模式) - PLCC: {finetuned_ava_plcc:.4f}, SRCC: {finetuned_ava_srcc:.4f}, MSE: {finetuned_ava_mse:.4f}")
        print(f"微调模型(BAID模式) - PLCC: {finetuned_baid_plcc:.4f}, SRCC: {finetuned_baid_srcc:.4f}, MSE: {finetuned_baid_mse:.4f}")
    
    # 4. 生成并返回报告
    print("\n综合评估报告:")
    if 'AVA' in results:
        print("在AVA数据集上:")
        ava_improvement_plcc = results['AVA']['finetuned_ava']['PLCC'] - results['AVA']['original']['PLCC']
        ava_improvement_srcc = results['AVA']['finetuned_ava']['SRCC'] - results['AVA']['original']['SRCC']
        print(f"  微调后PLCC提升: {ava_improvement_plcc:.4f} ({'+' if ava_improvement_plcc >= 0 else ''}{ava_improvement_plcc*100:.2f}%)")
        print(f"  微调后SRCC提升: {ava_improvement_srcc:.4f} ({'+' if ava_improvement_srcc >= 0 else ''}{ava_improvement_srcc*100:.2f}%)")
    
    if 'BAID' in results:
        print("在BAID数据集上:")
        baid_improvement_plcc = results['BAID']['finetuned_baid']['PLCC'] - results['BAID']['original']['PLCC']
        baid_improvement_srcc = results['BAID']['finetuned_baid']['SRCC'] - results['BAID']['original']['SRCC']
        print(f"  微调后PLCC提升: {baid_improvement_plcc:.4f} ({'+' if baid_improvement_plcc >= 0 else ''}{baid_improvement_plcc*100:.2f}%)")
        print(f"  微调后SRCC提升: {baid_improvement_srcc:.4f} ({'+' if baid_improvement_srcc >= 0 else ''}{baid_improvement_srcc*100:.2f}%)")
    
    # 可选：绘制散点图比较预测分数和真实分数
    try:
        import matplotlib.pyplot as plt

        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题        
        
        if 'BAID' in results:
            plt.figure(figsize=(15, 5))
            
            # 原始模型在BAID上的表现
            plt.subplot(1, 3, 1)
            plt.scatter(true_scores, original_pred_scores, alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.title(f'原始模型 (PLCC={original_plcc:.4f}, SRCC={original_srcc:.4f})')
            plt.xlabel('真实分数')
            plt.ylabel('预测分数')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            # 微调模型(AVA模式)在BAID上的表现
            plt.subplot(1, 3, 2)
            plt.scatter(true_scores, finetuned_ava_scores, alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.title(f'微调模型-AVA (PLCC={finetuned_ava_plcc:.4f}, SRCC={finetuned_ava_srcc:.4f})')
            plt.xlabel('真实分数')
            plt.ylabel('预测分数')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            # 微调模型(BAID模式)在BAID上的表现
            plt.subplot(1, 3, 3)
            plt.scatter(true_scores, finetuned_baid_scores, alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.title(f'微调模型-BAID (PLCC={finetuned_baid_plcc:.4f}, SRCC={finetuned_baid_srcc:.4f})')
            plt.xlabel('真实分数')
            plt.ylabel('预测分数')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig('baid_validation_comparison.png', dpi=300)
            plt.show()
    except Exception as e:
        print(f"绘图时出错: {str(e)}")
    
    return results


# 使用示例
if __name__ == "__main__":
    # 配置参数
    ava_pretrained_path = r"C:\Users\Administrator\Documents\GitHub\ReLIC\code\AVA\pretrain_model\relic2_model.pth"
    baid_data_path = r"D:\Datasets\BAID"
    baid_data_dir = r"D:\Datasets\BAID\validation"
    save_dir = r"C:\Users\Administrator\Documents\GitHub\ReLIC\code\AVA\finetune_models"
    
    # 执行微调或预测
    import argparse
    parser = argparse.ArgumentParser(description='NIMA模型微调与预测')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'predict', 'batch', 'validate'], 
                        help='运行模式: train (微调), predict (单图预测), batch (批量预测), validate (验证对比)')
    parser.add_argument('--ava_val_csv', type=str, 
                        default=os.path.join(option.init().path_to_save_csv, 'val.csv'),
                        help='AVA验证集CSV路径')
    parser.add_argument('--baid_data_path', type=str, 
                    default=baid_data_path, 
                    help='BAID数据集根目录路径')
    parser.add_argument('--model_path', type=str, default=os.path.join(save_dir, "best_model.pth"),
                        help='预测时使用的模型路径')
    parser.add_argument('--original_model_path', type=str, 
                    default=ava_pretrained_path, 
                    help='原始NIMA模型路径')
    parser.add_argument('--image_path', type=str, 
                        default=r'C:\Users\Administrator\Documents\GitHub\ReLIC\Images\69.jpg', 
                        help='预测单张图像的路径')
    parser.add_argument('--image_dir', type=str, 
                        help='批量预测的图像目录')
    parser.add_argument('--output_csv', type=str, 
                        help='批量预测结果的CSV输出路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 执行微调
        finetune_on_baid(
            pretrained_path=ava_pretrained_path,
            baid_data_path=baid_data_path,   
            save_dir=save_dir,
            num_epochs=35,
            batch_size=32,
            lr = 1e-4,
            if_continue=False
        )
    elif args.mode == 'validate':
        # 验证对比
        validate_models(
            original_model_path=args.original_model_path,
            finetuned_model_path=args.model_path,
            ava_val_csv=args.ava_val_csv,
            baid_data_dir=args.baid_data_path  # 直接使用BAID数据集根目录
        )
    elif args.mode == 'predict':
        # 单图预测
        if not args.image_path:
            print("请指定要预测的图像路径 (--image_path)")
        else:
            pred_single_finetuned(args.model_path, args.image_path,
                                  original_model_path=args.original_model_path)
    elif args.mode == 'batch':
        # 批量预测
        if not args.image_dir:
            print("请指定要预测的图像目录 (--image_dir)")
        else:
            batch_predict(
                model_path=args.model_path, 
                image_dir=args.image_dir, 
                output_csv=args.output_csv, 
                threshold=0.6
            )