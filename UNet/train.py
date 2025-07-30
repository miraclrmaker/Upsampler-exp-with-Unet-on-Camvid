import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch.optim.lr_scheduler import Expone
from tqdm import tqdm

import pandas as pd
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
#更改你的数据集目录
# work_path = os.path.dirname(os.path.abspath(__file__))
work_path = './'
dir_img = work_path + "CamVid/train/"
dir_mask = work_path + "CamVid/train_labels/"
dir_val_img = work_path + "CamVid/val/"
dir_val_mask = work_path + "CamVid/val_labels/"
dir_checkpoint = Path('./checkpoints/')

def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
#保证全局各个随机种子一致，便于实验复现

def find_latest_checkpoint(style1, style2, checkpoint_dir='./checkpoints/'):
    """查找指定算子组合的最新checkpoint"""
    style_dir = Path(checkpoint_dir) / f"{style1}_{style2}"
    if not style_dir.exists():
        return None
    
    # 查找所有checkpoint文件
    checkpoints = list(style_dir.glob('checkpoint_epoch*.pth'))
    if not checkpoints:
        return None
    
    # 按epoch数排序，返回最新的
    def extract_epoch(path):
        try:
            return int(path.stem.split('epoch')[1])
        except:
            return 0
    
    latest_checkpoint = max(checkpoints, key=extract_epoch)
    logging.info(f'Found latest checkpoint: {latest_checkpoint}')
    return str(latest_checkpoint)

def cleanup_gpu_memory():
    """清理GPU内存碎片"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 强制垃圾回收
        import gc
        gc.collect()
        # 记录内存使用情况
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f'GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB')

def setup_memory_optimization():
    """设置内存优化配置"""
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 设置其他内存优化选项
    if torch.cuda.is_available():
        # 启用内存池
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 保持setup_seed中的benchmark=False设置，不在这里修改
        
        logging.info('Memory optimization settings applied')

# 创建tensorboard日志目录
log_dir = Path('./logs/tensorboard')
log_dir.mkdir(parents=True, exist_ok=True)


def train_model(
        seed,
        model,
        device,
        epochs: int = 10,
        batch_size: int = 1,
        learning_rate: float = 3e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        normalize_input: bool = True,   
        weight_strategy: str = 'log', 
        resume_checkpoint: str = None,  
):
    # 创建tensorboard writer，按算子组合分组
    tb_log_dir = f'./logs/tensorboard/{model.style1}_{model.style2}/seed_{seed}'
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # 使用标准的CamVid分割，根据参数决定是否启用数据增强
    train_dataset = BasicDataset(dir_img, dir_mask, img_scale, normalize=normalize_input, augment=args.augment)
    val_dataset = BasicDataset(dir_val_img, dir_val_mask, img_scale, normalize=normalize_input, augment=False)


    train_set = train_dataset
    val_set = val_dataset
    n_train = len(train_set)
    n_val = len(val_set)

    # 数据加载
    num_workers = min(16, os.cpu_count())  # 限制最大工作进程数
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Input normalize: {normalize_input}
        Data augmentation: {args.augment}
        Mixed Precision: {amp}
    ''')
    
    # 计算类别权重 - 对于不平衡的CamVid数据集很重要
    class_weights = None
    if hasattr(train_dataset, 'mask_values') and len(train_dataset.mask_values) > 1:
        # 使用缓存机制快速计算权重
        class_weights = compute_class_weights_cached(train_dataset, device, weight_strategy=weight_strategy, force_recompute=args.force_recompute_weights)
        logging.info(f"Class weights: {class_weights}")
        
        # 显示类别名称对应的权重，便于理解
        class_names = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                      'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
        for i, (name, weight) in enumerate(zip(class_names, class_weights)):
            logging.info(f"Class {i} ({name}): weight={weight:.4f}")
    
    optimizer = optim.Adam(model.parameters(),
                          lr=learning_rate, 
                          weight_decay=weight_decay,
                          betas=(0.9, 0.999),
                          eps=1e-8)
    
    # WarmupPoly学习率调度策略
    class WarmupPolyLR:
        def __init__(self, optimizer, max_epochs, warmup_epochs=5, power=0.9, min_lr=1e-6):
            self.optimizer = optimizer
            self.max_epochs = max_epochs
            self.warmup_epochs = warmup_epochs
            self.power = power
            self.min_lr = min_lr
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
            
        def step(self, epoch):
            if epoch < self.warmup_epochs:
                # Warmup阶段：线性增长
                lr_scale = epoch / self.warmup_epochs
                lrs = [base_lr * lr_scale for base_lr in self.base_lrs]
            else:
                # Poly decay阶段：多项式衰减
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                lr_scale = (1 - progress) ** self.power
                lrs = [max(base_lr * lr_scale, self.min_lr) for base_lr in self.base_lrs]
            
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
            
            return lrs[0]  
    # 使用WarmupPoly调度器
    scheduler = WarmupPolyLR(
        optimizer, 
        max_epochs=epochs,
        warmup_epochs=5,      # 前5个epoch进行warmup
        power=0.9,            # poly decay的指数，0.9是常用值
        min_lr=1e-6           # 最小学习率
    )
    
    # 备选：如果想使用ReduceLROnPlateau，可以注释掉上面的WarmupPoly，启用下面的代码
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='max',           # 监控验证指标的最大值
    #     patience=3,           # 更短的耐心：3轮不改善就降低学习率
    #     factor=0.6,           # 更激进的衰减：学习率减半
    #     min_lr=1e-7,          # 最小学习率
    #     verbose=True
    # )
    
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)

    # 使用ignore_index=255来忽略unlabelled像素，现在模型只有11个输出类别
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_score = 0
    start_epoch = 1

    # 断点续训逻辑
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logging.info(f'Loading checkpoint from {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容旧格式的checkpoint
            state_dict = checkpoint.copy()
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
            model.load_state_dict(state_dict)
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info('Optimizer state loaded')
        
        # 加载训练状态
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f'Resuming from epoch {start_epoch}')
        
        if 'best_score' in checkpoint:
            best_score = checkpoint['best_score']
            logging.info(f'Best score so far: {best_score}')
        
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            logging.info(f'Global step: {global_step}')
        
        # 加载学习率调度器状态（如果有的话）
        if hasattr(scheduler, 'load_state_dict') and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info('Scheduler state loaded')
            except:
                logging.warning('Failed to load scheduler state, using default')
        
        # 加载GradScaler状态
        if 'scaler_state_dict' in checkpoint:
            grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logging.info('GradScaler state loaded')
        
        logging.info(f'Successfully resumed from {resume_checkpoint}')
    else:
        if resume_checkpoint:
            logging.warning(f'Checkpoint {resume_checkpoint} not found, starting from scratch')

    # Begin training
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        temp = 0
        
        # 每10个epoch进行一次深度内存清理（减少频率）
        if epoch % 10 == 0:
            cleanup_gpu_memory()
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # 主要的交叉熵损失（已经处理ignore_index=255）
                        loss = criterion(masks_pred, true_masks)
                        
                        # 处理dice loss时需要排除ignore_index=255的像素
                        valid_mask = (true_masks != 255)
                        if valid_mask.sum() > 0:
                            # 只对有效像素计算dice loss
                            masks_pred_valid = masks_pred.clone()
                            true_masks_valid = true_masks.clone()
                            
                            # 将ignore像素临时设为0类别用于one-hot编码
                            true_masks_valid[true_masks == 255] = 0
                            
                            # 创建one-hot编码
                            true_masks_oh = F.one_hot(true_masks_valid, model.n_classes).permute(0, 3, 1, 2).float()
                            
                            # 将ignore像素的one-hot设为0
                            valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(true_masks_oh)
                            true_masks_oh = true_masks_oh * valid_mask_expanded.float()
                            
                            # 对预测结果也应用valid_mask
                            masks_pred_softmax = F.softmax(masks_pred_valid, dim=1)
                            masks_pred_softmax = masks_pred_softmax * valid_mask_expanded.float()
                            
                            dice_loss_val = dice_loss(
                                masks_pred_softmax,
                                true_masks_oh,
                                multiclass=True
                            )
                            loss += dice_loss_val

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                # 每100步记录一次训练损失到tensorboard，减少I/O开销
                if global_step % 100 == 0:
                    writer.add_scalar('Train/Loss', epoch_loss / (global_step % 100 + 1), global_step)
                    writer.add_scalar('Train/Epoch', epoch, global_step)
                    writer.add_scalar('Train/Seed', seed, global_step)
                
                pbar.set_postfix(**{'loss (batch)': epoch_loss / (global_step - (epoch-start_epoch) * len(train_loader) + len(train_loader))})

        # 每个epoch结束后进行一次验证
        # 每5个epoch记录一次模型权重直方图到tensorboard，减少I/O开销
        if epoch % 5 == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    writer.add_histogram(f'Weights/{tag}', value.data.cpu(), epoch)

        val_dice, val_miou, val_per_class_acc, val_per_class_iou = evaluate(model, val_loader, device, amp)
        if val_dice > best_score:
            best_score = val_dice
        if temp < val_dice:
            temp = val_dice

        
        # 调用WarmupPoly学习率调度器
        current_lr = scheduler.step(epoch)  # WarmupPoly返回当前学习率
        
        # 备选：如果使用ReduceLROnPlateau，取消注释下面的代码并注释上面的代码
        # scheduler.step(val_dice)  # 传入要监控的验证指标
        # current_lr = optimizer.param_groups[0]['lr']  # 手动获取当前学习率

        # 记录当前学习率到日志
        logging.info(f'Epoch {epoch}/{epochs} - Learning Rate: {current_lr:.2e}')

        # 计算平均每类像素精度用于日志记录
        mean_per_class_acc = sum(val_per_class_acc) / len(val_per_class_acc)
        logging.info(f'Epoch {epoch}/{epochs} - Validation Dice: {val_dice:.4f}, mIoU: {val_miou:.4f}, mean per-class acc: {mean_per_class_acc:.4f}')
        
        # 详细的每类别像素精度日志（可选，用于调试）
        class_names = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                      'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
        class_acc_info = ", ".join([f"{class_names[i]}: {acc:.3f}" for i, acc in enumerate(val_per_class_acc)])
        logging.info(f'Per-class accuracies - {class_acc_info}')
        
        # 详细的每类别IoU日志
        class_iou_info = ", ".join([f"{class_names[i]}: {iou:.3f}" for i, iou in enumerate(val_per_class_iou)])
        logging.info(f'Per-class IoU - {class_iou_info}')
        try:
            # CamVid类别名称（11个有效类别，不包括unlabelled）
            class_names = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                          'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
            
            # 记录学习率和验证指标到tensorboard
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Validation/Dice', val_dice, epoch)
            writer.add_scalar('Validation/mIoU', val_miou, epoch)
            writer.add_scalar('Validation/mean_per_class_acc', mean_per_class_acc, epoch)
            
            # 为每个类别创建详细的像素精度记录
            for i, (class_name, acc) in enumerate(zip(class_names, val_per_class_acc)):
                writer.add_scalar(f'Validation/per_class_acc_{class_name}', acc, epoch)
            
            # 为每个类别创建详细的IoU记录
            for i, (class_name, iou) in enumerate(zip(class_names, val_per_class_iou)):
                writer.add_scalar(f'Validation/per_class_iou_{class_name}', iou, epoch)
            
            # 每10个epoch记录一次图像到tensorboard，减少存储开销
            if epoch % 10 == 0:
                writer.add_image('Images/Input', images[0], epoch)
                writer.add_image('Images/True_Mask', true_masks[0].float().unsqueeze(0), epoch)
                writer.add_image('Images/Pred_Mask', masks_pred.argmax(dim=1)[0].float().unsqueeze(0), epoch)
            
        except Exception as e:
            print('tensorboard log error:', e)

        # 记录epoch级别的指标到tensorboard
        writer.add_scalar('Epoch/Seed', seed, epoch)
        writer.add_scalar('Epoch/Validation_Dice', temp.cpu().item(), epoch)
        writer.add_scalar('Epoch/Validation_mIoU', val_miou, epoch)
        writer.add_scalar('Epoch/Validation_mean_per_class_acc', mean_per_class_acc, epoch)
        writer.add_scalar('Epoch/Global_Step', global_step, epoch)
        # lis.append(temp.cpu().item())

        if save_checkpoint:
            # 为不同的算子组合创建专门的文件夹
            style_dir_name = f"{model.style1}_{model.style2}"
            style_checkpoint_dir = dir_checkpoint / style_dir_name
            Path(style_checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            # 保存完整的训练状态
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': grad_scaler.state_dict(),
                'best_score': best_score,
                'global_step': global_step,
                'mask_values': train_dataset.mask_values,
                'val_dice': val_dice,
                'val_miou': val_miou,
                'seed': seed,
                'args': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'img_scale': img_scale,
                    'normalize_input': normalize_input,
                    'weight_strategy': weight_strategy
                }
            }
            
            # 保存调度器状态（如果支持的话）
            if hasattr(scheduler, 'state_dict'):
                try:
                    checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()
                except:
                    pass
            
            checkpoint_path = style_checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
            torch.save(checkpoint_state, str(checkpoint_path))
            
            # 保存最佳模型
            if val_dice > best_score - 1e-6:  # 使用小的容差避免浮点精度问题
                best_path = style_checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint_state, str(best_path))
                logging.info(f'Best model saved to {best_path}!')
            
            logging.info(f'Checkpoint {epoch} saved to {checkpoint_path}!')
            
            # 清理旧的checkpoint以节省空间（保留最近10个）
            if epoch > 10:
                old_checkpoint = style_checkpoint_dir / f'checkpoint_epoch{epoch-10}.pth'
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logging.info(f'Removed old checkpoint: {old_checkpoint}')



    # 记录最佳分数到tensorboard
    writer.add_scalar('Final/Best_Score', best_score, epochs)
    writer.add_scalar('Final/Seed', seed, epochs)
    
    # 关闭tensorboard writer
    writer.close()
    
    print(".................................", best_score, ".....................................")
    return best_score, train_dataset

def compute_class_weights_cached(train_dataset, device, cache_file='class_weights_cache.pt', weight_strategy='log', force_recompute=False):
    """计算类别权重，支持缓存机制
    Args:
        train_dataset: 训练数据集
        device: 计算设备
        cache_file: 缓存文件路径
        weight_strategy: 权重计算策略，'log' 或 'median_frequency'
    """
    import hashlib
    
    # 创建数据集特征哈希来检查缓存是否有效
    # 注意：对于有数据增强的数据集，我们仍然使用相同的权重
    dataset_info = f"{len(train_dataset)}_{train_dataset.scale}_{train_dataset.normalize}_{len(train_dataset.mask_values)}_{weight_strategy}"
    cache_key = hashlib.md5(dataset_info.encode()).hexdigest()
    
    # 尝试加载缓存（除非强制重新计算）
    if not force_recompute and os.path.exists(cache_file):
        try:
            cached_data = torch.load(cache_file, map_location='cpu')
            if cached_data.get('cache_key') == cache_key:
                logging.info(f"Using cached class weights (strategy: {weight_strategy})")
                return cached_data['class_weights'].to(device)
            else:
                logging.info(f"Cache key mismatch (likely different strategy), recomputing weights...")
        except Exception as e:
            logging.info(f"Cache file corrupted, recomputing... Error: {e}")
    elif force_recompute:
        logging.info("Forcing recomputation of class weights (ignoring cache)...")
    else:
        logging.info("No cache file found, computing class weights...")
    
    # 重新计算权重 - 临时禁用数据增强以获得真实的类别分布
    original_augment = train_dataset.augment
    train_dataset.augment = False  # 临时禁用数据增强
    
    logging.info(f"Using weight strategy: {weight_strategy}")
    logging.info(f"Computing class weights from all training data (without augmentation)...")
    from collections import Counter
    
    pixel_counts = Counter()
    total_pixels = 0
    
    with tqdm(total=len(train_dataset), desc="Computing class weights", unit="sample") as pbar:
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            mask = sample['mask']
            unique, counts = torch.unique(mask, return_counts=True)
            for cls, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                pixel_counts[cls] += count
                total_pixels += count
            pbar.update(1)
    
    # 恢复原始数据增强设置
    train_dataset.augment = original_augment
    
    # 获取类别数量
    n_classes = len(train_dataset.mask_values)
    
    # 计算每个类别的频率（只考虑有效类别，排除255）
    frequencies = torch.zeros(n_classes, dtype=torch.float32, device='cpu')  # 先在CPU上计算
    valid_total_pixels = total_pixels - pixel_counts.get(255, 0)
    
    logging.info(f"Total pixels: {total_pixels}, Valid pixels: {valid_total_pixels}")
    logging.info(f"Pixel counts: {dict(pixel_counts)}")
    
    for cls in range(n_classes):
        if cls in pixel_counts and pixel_counts[cls] > 0:
            frequencies[cls] = pixel_counts[cls] / valid_total_pixels
        else:
            frequencies[cls] = 1e-8
    
    # 选择权重计算策略
    if weight_strategy == 'log':
        # 使用基于对数的权重计算策略（更适合严重不平衡的数据集）
        # 公式: weight[i] = 1 / log(normVal + frequency[i])
        # 其中 normVal=1.10 是归一化常数，防止log接近0
        import numpy as np
        norm_val = 1.10
        
        # 将频率转换为numpy数组便于计算
        frequencies_np = frequencies.cpu().numpy()
        
        # 使用对数权重计算
        weights_np = np.zeros(n_classes, dtype=np.float32)
        for i in range(n_classes):
            if frequencies_np[i] > 0:
                # 基于对数的权重计算
                weights_np[i] = 1.0 / np.log(norm_val + frequencies_np[i])
            else:
                # 对于频率为0的类别，给予较高权重
                weights_np[i] = 1.0 / np.log(norm_val + 1e-8)
        
        # 转换回torch tensor
        weights = torch.from_numpy(weights_np)
        
        # 归一化权重，使平均权重为1
        weights = weights / weights.mean()
        
        # 限制权重范围防止过大或过小的权重
        weights = torch.clamp(weights, min=0.1, max=20.0)
        
        logging.info(f"Log-based class weights computed (normVal={norm_val}):")
        class_names = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                      'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
        for i, (name, weight, freq) in enumerate(zip(class_names, weights, frequencies)):
            logging.info(f"  {name}: weight={weight:.4f}, frequency={freq:.6f}")
    
    elif weight_strategy == 'median_frequency':
        non_zero_freqs = frequencies[frequencies > 1e-7]
        if len(non_zero_freqs) > 0:
            median_freq = torch.median(non_zero_freqs)
            weights = median_freq / (frequencies + 1e-8)
        else:
            weights = torch.ones(n_classes, dtype=torch.float32)
        
        # 限制权重范围
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        logging.info(f"Median frequency class weights computed:")
        class_names = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                      'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
        for i, (name, weight, freq) in enumerate(zip(class_names, weights, frequencies)):
            logging.info(f"  {name}: weight={weight:.4f}, frequency={freq:.6f}")
    
    else:
        raise ValueError(f"Unknown weight strategy: {weight_strategy}. Use 'log' or 'median_frequency'")
    
    weights = weights.to(device)
    
    torch.save({
        'cache_key': cache_key,
        'class_weights': weights.cpu(), 
        'frequencies': frequencies
    }, cache_file)
    logging.info(f"Class weights cached to {cache_file}")
    
    return weights

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=120, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=11, help='Number of classes (excluding unlabelled)')
    parser.add_argument('--normalize', action='store_true', default=True, help='Use ImageNet normalization on input images')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', help='Disable ImageNet normalization on input images')
    parser.add_argument('--augment', action='store_true', default=True, help='Enable data augmentation (random scaling, flipping, cropping)')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='Disable data augmentation')
    parser.add_argument('--compile', action='store_true', default=False, help='Compile model with torch.compile for faster training')
    parser.add_argument('--channels-last', action='store_true', default=True, help='Use channels last memory format')
    parser.add_argument('--weight-strategy', type=str, default='log', choices=['log', 'median_frequency'], 
                        help='Class weight computation strategy: log (1/log(1.1+freq)) or median_frequency')
    parser.add_argument('--force-recompute-weights', action='store_true', default=False,
                        help='Force recomputation of class weights, ignoring cache')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    parser.add_argument('--auto-resume', action='store_true', default=False,
                        help='Automatically resume from the latest checkpoint for each algorithm combination')

    return parser.parse_args()


if __name__ == '__main__':
    for i in range(1):
        setup_seed(i)
        args = get_args()
        
        # 设置内存优化
        setup_memory_optimization()

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        
        
        cleanup_gpu_memory()
        # device = [torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel

        # if args.load:
        #     state_dict = torch.load(args.load, map_location=device)
        #     del state_dict['mask_values']
        #     model.load_state_dict(state_dict)
        #     logging.info(f'Model loaded from {args.load}')

        try:
        #TODO 选择你的采样算子
            down = ['max-pooling']
            up = ['dysample_lp-dynamic']
            dic = {} 
            for do in down:
                for u in up:
                    logging.info(f"Starting experiment with down-sampling: {do}, up-sampling: {u}")
                    
                    
                    cleanup_gpu_memory()
                    
                    model = UNet(n_channels=3, n_classes=args.classes, style1=do, style2=u)
                    # model = nn.DataParallel(model, device_ids=device)
                    
                    # 使用channels last内存格式优化
                    if args.channels_last:
                        model = model.to(memory_format=torch.channels_last)
                    
                    logging.info(f'Network ({do}_{u}):\n'
                                 f'\t{model.n_channels} input channels\n'
                                 f'\t{model.n_classes} output channels (classes)\n'
                                 f'\tDown-sampling: {model.style1}\n'
                                 f'\tUp-sampling: {model.style2}')
                    model.to(device=device)
                    
                    # 编译模型以获得更好的性能 (PyTorch 2.0+)
                    if args.compile and hasattr(torch, 'compile'):
                        logging.info("Compiling model with torch.compile...")
                        model = torch.compile(model)
                    
                    # 确定断点续训路径
                    resume_checkpoint = None
                    if args.resume:
                        resume_checkpoint = args.resume
                    elif args.auto_resume:
                        resume_checkpoint = find_latest_checkpoint(do, u)
                    
                    if resume_checkpoint:
                        logging.info(f"Will attempt to resume from: {resume_checkpoint}")
                    
                    best_score, _ = train_model(
                        seed=i,
                        model=model,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                        device=device,
                        img_scale=args.scale,
                        val_percent=args.val / 100,
                        amp=args.amp,
                        normalize_input=args.normalize,
                        weight_strategy=args.weight_strategy,
                        resume_checkpoint=resume_checkpoint
                    )
                    
                    # 存储每种算子组合的结果
                    dic[f"{do}_{u}"] = best_score
                    logging.info(f"Experiment {do}_{u} completed with best score: {best_score}")
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                          'Enabling checkpointing to reduce memory usage, but this slows down training. '
                          'Consider enabling AMP (--amp) for fast and memory efficient training')
            
            # 深度内存清理
            cleanup_gpu_memory()
            
            # 尝试启用checkpointing（需要修复实现）
            try:
                model.use_checkpointing()
            except:
                logging.warning("Checkpointing not available, continuing without it")
            
            # 确定断点续训路径
            resume_checkpoint = None
            if args.resume:
                resume_checkpoint = args.resume
            elif args.auto_resume:
                resume_checkpoint = find_latest_checkpoint(do, u)
            
            best_score, _ = train_model(
                seed=i,
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                normalize_input=args.normalize,
                weight_strategy=args.weight_strategy,
                resume_checkpoint=resume_checkpoint
            )
            
            # 存储每种算子组合的结果
            dic[f"{do}_{u}"] = best_score
            logging.info(f"Experiment {do}_{u} completed with best score: {best_score} (after OOM recovery)")
            
        # 输出所有实验结果汇总
        logging.info("="*50)
        logging.info("Experiment Results Summary:")
        for combo, score in dic.items():
            logging.info(f"  {combo}: {score:.6f}")
        logging.info("="*50)
        
        # 保存实验结果到CSV文件
        import pandas as pd
        results_df = pd.DataFrame([
            {'seed': i, 'algorithm_combo': combo, 'best_dice_score': score}
            for combo, score in dic.items()
        ])
        results_file = f'experiment_results_seed_{i}.csv'
        results_df.to_csv(results_file, index=False)
        logging.info(f"Experiment results saved to {results_file}")

        # file_path = os.path.dirname(os.path.realpath(__file__))
        # file_path = os.path.join(file_path, '..')
        # name = file_path + "/seed" + str(i) + '.csv'
        # if os.path.exists(name):
        #     df.to_csv(name, mode='a', index=False)
        # else:
        #     df.to_csv(name, index=False)
