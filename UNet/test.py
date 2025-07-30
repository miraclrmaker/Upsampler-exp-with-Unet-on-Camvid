#!/usr/bin/env python3
"""
UNet模型测试脚本 - 使用测试集选择最佳模型
"""
import argparse
import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict

from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff

    # 测试集路径
work_path = './'
dir_test_img = work_path + "CamVid/test/"  
dir_test_mask = work_path + "CamVid/test_labels/"  
dir_checkpoint = Path('./checkpoints/')  

# CamVid类别名称
CLASS_NAMES = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
               'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']

def find_best_checkpoint_from_last_n(checkpoint_dir, n_models=10, device='cuda'):
    """从最后n个checkpoint中找到验证集上表现最好的模型"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # 检查是否是算子组合的子文件夹结构
    style_combo = None
    if checkpoint_dir.name != 'checkpoints':
        # 如果是子文件夹，提取算子组合名称
        style_combo = checkpoint_dir.name
        logging.info(f"Detected algorithm combination: {style_combo}")
    
    # 获取所有checkpoint文件，按epoch编号排序
    checkpoint_files = []
    for file in checkpoint_dir.glob('checkpoint_epoch*.pth'):
        try:
            epoch_num = int(file.stem.split('epoch')[1])
            checkpoint_files.append((epoch_num, file))
        except:
            continue
    
    # 按epoch编号排序，取最后n个
    checkpoint_files.sort(key=lambda x: x[0])
    last_n_checkpoints = checkpoint_files[-n_models:] if len(checkpoint_files) >= n_models else checkpoint_files
    
    if not last_n_checkpoints:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}")
    
    logging.info(f"Found {len(last_n_checkpoints)} checkpoints to evaluate")
    logging.info(f"Epoch range: {last_n_checkpoints[0][0]} - {last_n_checkpoints[-1][0]}")
    
    return last_n_checkpoints, style_combo
def evaluate_best_checkpoint(last_n_checkpoints, device, style_combo=None):
    """评估多个checkpoint并返回最佳的一个 - 基于测试集结果"""
    # 创建测试数据集来评估每个checkpoint
    test_dataset = BasicDataset(
        dir_test_img,  # 直接使用测试集
        dir_test_mask,  # 直接使用测试集标签
        scale=1.0, 
        normalize=True, 
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    best_miou = -1
    best_checkpoint = None
    best_epoch = -1
    best_style1, best_style2 = 'max-pooling', 'bilinear'
    
    # 从算子组合名称推断模型参数
    if style_combo:
        style_parts = style_combo.split('_')
        if len(style_parts) >= 2:
            style1 = style_parts[0]
            style2 = '_'.join(style_parts[1:])  # 支持多段式的算子名称如dysample_lp
        else:
            # 默认参数
            style1, style2 = 'max-pooling', 'bilinear'
    else:
        style1, style2 = 'max-pooling', 'bilinear'
    
    logging.info(f"Using model configuration: style1={style1}, style2={style2}")
    
    # 评估每个checkpoint
    for epoch, checkpoint_file in tqdm(last_n_checkpoints, desc="Evaluating checkpoints"):
        try:
            # 加载模型 - 使用从算子组合推断的参数
            model = UNet(n_channels=3, n_classes=11, style1=style1, style2=style2)
            
            # 使用兼容的checkpoint加载
            model_state_dict, mask_values, checkpoint_info = load_checkpoint_compatible(checkpoint_file, device)
            model.load_state_dict(model_state_dict)
            model.to(device)
            model.eval()
            
            # 在测试集上评估
            miou = evaluate_model_miou(model, test_loader, device)
            
            logging.info(f"Epoch {epoch}: mIoU = {miou:.4f}")
            
            if miou > best_miou:
                best_miou = miou
                best_checkpoint = checkpoint_file
                best_epoch = epoch
                best_style1, best_style2 = style1, style2  # 保存最佳模型的配置
                
        except Exception as e:
            logging.warning(f"Failed to evaluate checkpoint {checkpoint_file}: {e}")
            continue
    
    if best_checkpoint is None:
        raise RuntimeError("No valid checkpoint could be evaluated")
    
    logging.info(f"Best checkpoint: Epoch {best_epoch} with Test mIoU {best_miou:.4f}")
    logging.info(f"Best checkpoint file: {best_checkpoint}")
    
    return best_checkpoint, best_epoch, best_miou, (best_style1, best_style2)

def evaluate_model_miou(model, dataloader, device):
    """快速评估模型的mIoU（基于测试集选择最佳checkpoint）"""
    model.eval()
    total_inter = torch.zeros(11, device=device)
    total_union = torch.zeros(11, device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            images, masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long)
            
            with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
                outputs = model(images)
                preds = outputs.argmax(dim=1)
            
            # 计算IoU
            for cls in range(11):
                pred_i = (preds == cls)
                mask_i = (masks == cls)
                valid_mask = (masks != 255)  # 排除ignore像素
                
                pred_i = pred_i & valid_mask
                mask_i = mask_i & valid_mask
                
                inter = (pred_i & mask_i).sum()
                union = (pred_i | mask_i).sum()
                
                total_inter[cls] += inter
                total_union[cls] += union
    
    iou = total_inter / (total_union + 1e-6)
    miou = iou.mean().item()
    return miou

@torch.inference_mode()
def comprehensive_test_evaluation(model, test_loader, device, amp=True):
    """在测试集上进行全面评估"""
    model.eval()
    num_test_batches = len(test_loader)
    n_classes = model.n_classes
    
    # 初始化统计变量
    dice_score = 0
    total_inter = torch.zeros(n_classes, device=device)
    total_union = torch.zeros(n_classes, device=device)
    total_correct = torch.zeros(n_classes, device=device)
    total_label = torch.zeros(n_classes, device=device)
    total_pixels = 0
    correct_pixels = 0
    
    # 混淆矩阵
    confusion_matrix = torch.zeros(n_classes, n_classes, device=device)
    
    logging.info("Starting comprehensive evaluation on test set...")
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(test_loader, total=num_test_batches, desc='Testing', unit='batch'):
            images, masks_true = batch['image'], batch['mask']
            
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = masks_true.to(device=device, dtype=torch.long)
            
            # 预测
            masks_pred = model(images)
            pred = masks_pred.argmax(dim=1)
            
            # Dice Score计算
            valid_mask = (masks_true != 255)
            if valid_mask.sum() > 0:
                temp_true = masks_true.clone()
                temp_pred = pred.clone()
                temp_true[masks_true == 255] = 0
                temp_pred[masks_true == 255] = 0
                
                mask_true_oh = F.one_hot(temp_true, n_classes).permute(0, 3, 1, 2).float()
                mask_pred_oh = F.one_hot(temp_pred, n_classes).permute(0, 3, 1, 2).float()
                
                valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(mask_true_oh)
                mask_true_oh = mask_true_oh * valid_mask_expanded.float()
                mask_pred_oh = mask_pred_oh * valid_mask_expanded.float()
                
                dice_score += multiclass_dice_coeff(mask_pred_oh, mask_true_oh, reduce_batch_first=False)
            
            # IoU和准确度计算
            pred_flat = pred.view(-1)
            label_flat = masks_true.view(-1)
            
            # 只计算有效像素（排除ignore_index=255）
            valid_pixels_mask = (label_flat != 255)
            if valid_pixels_mask.sum() > 0:
                valid_pred = pred_flat[valid_pixels_mask]
                valid_label = label_flat[valid_pixels_mask]
                
                # 总体像素准确度
                total_pixels += valid_pixels_mask.sum().item()
                correct_pixels += (valid_pred == valid_label).sum().item()
                
                # 每类别统计
                for cls in range(n_classes):
                    pred_i = (valid_pred == cls)
                    label_i = (valid_label == cls)
                    
                    inter = (pred_i & label_i).sum()
                    union = (pred_i | label_i).sum()
                    
                    total_inter[cls] += inter
                    total_union[cls] += union
                    total_correct[cls] += inter
                    total_label[cls] += label_i.sum()
                    
                    # 更新混淆矩阵
                    for pred_cls in range(n_classes):
                        confusion_matrix[cls, pred_cls] += ((valid_label == cls) & (valid_pred == pred_cls)).sum()
    
    # 计算最终指标
    mean_dice = dice_score / max(num_test_batches, 1)
    
    # 总体像素准确度
    overall_pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    # IoU和mIoU
    iou = total_inter / (total_union + 1e-6)
    miou = iou.mean().item()
    
    # 每类别像素准确度
    per_class_acc = (total_correct / (total_label + 1e-6)).cpu().numpy()
    
    # 每类别IoU
    per_class_iou = iou.cpu().numpy()
    
    # 频率权重IoU (Frequency Weighted IoU)
    freq = total_label / total_label.sum()
    fwiou = (freq * iou).sum().item()
    
    return {
        'mean_dice': mean_dice.item(),
        'overall_pixel_accuracy': overall_pixel_acc,
        'mean_iou': miou,
        'frequency_weighted_iou': fwiou,
        'per_class_accuracy': per_class_acc,
        'per_class_iou': per_class_iou,
        'confusion_matrix': confusion_matrix.cpu().numpy()
    }

def print_detailed_results(results, class_names):
    """打印详细的评估结果"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SET EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOVERALL METRICS:")
    print(f"   Mean Dice Score:              {results['mean_dice']:.4f}")
    print(f"   Overall Pixel Accuracy:       {results['overall_pixel_accuracy']:.4f}")
    print(f"   Mean IoU (mIoU):              {results['mean_iou']:.4f}")
    print(f"   Frequency Weighted IoU:       {results['frequency_weighted_iou']:.4f}")
    
    print(f"\nPER-CLASS METRICS:")
    print(f"{'Class':<12} {'Pixel Acc':<12} {'IoU':<12}")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12} {results['per_class_accuracy'][i]:<12.4f} {results['per_class_iou'][i]:<12.4f}")
    
    # 找出表现最好和最差的类别
    best_iou_idx = np.argmax(results['per_class_iou'])
    worst_iou_idx = np.argmin(results['per_class_iou'])
    
    print(f"\nBEST PERFORMING CLASS:")
    print(f"   {class_names[best_iou_idx]}: IoU = {results['per_class_iou'][best_iou_idx]:.4f}")
    print(f"\nWORST PERFORMING CLASS:")
    print(f"   {class_names[worst_iou_idx]}: IoU = {results['per_class_iou'][worst_iou_idx]:.4f}")

def save_results_to_csv(results, class_names, save_path, style_combo=None):
    """保存结果到CSV文件，支持算子组合分组"""
    # 如果指定了算子组合，创建对应的子文件夹
    if style_combo:
        save_path = Path(save_path) / style_combo
        logging.info(f"Saving results for algorithm combination: {style_combo}")
    else:
        save_path = Path(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 总体指标
    overall_metrics = {
        'Metric': ['Mean Dice', 'Overall Pixel Accuracy', 'Mean IoU', 'Frequency Weighted IoU'],
        'Value': [results['mean_dice'], results['overall_pixel_accuracy'], 
                 results['mean_iou'], results['frequency_weighted_iou']]
    }
    
    # 每类别指标
    per_class_metrics = {
        'Class': class_names,
        'Pixel_Accuracy': results['per_class_accuracy'],
        'IoU': results['per_class_iou']
    }
    
    # 保存文件
    pd.DataFrame(overall_metrics).to_csv(save_path / 'overall_metrics.csv', index=False)
    pd.DataFrame(per_class_metrics).to_csv(save_path / 'per_class_metrics.csv', index=False)
    
    # 保存混淆矩阵
    confusion_df = pd.DataFrame(results['confusion_matrix'], 
                               index=[f'True_{name}' for name in class_names],
                               columns=[f'Pred_{name}' for name in class_names])
    confusion_df.to_csv(save_path / 'confusion_matrix.csv')
    
    logging.info(f"Results saved to {save_path}")

def get_args():
    parser = argparse.ArgumentParser(description='Test UNet on CamVid test set')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--n-models', type=int, default=10, help='Number of last models to consider for best selection (based on test set performance)')
    parser.add_argument('--normalize', action='store_true', default=True, help='Use ImageNet normalization')
    parser.add_argument('--save-results', type=str, default='./test_results/', help='Directory to save results')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/', help='Directory containing checkpoints')
    parser.add_argument('--specific-combo', type=str, default=None, 
                        help='Test only a specific algorithm combination (e.g., "max-pooling_bilinear")')

    return parser.parse_args()

def load_checkpoint_compatible(checkpoint_path, device):
    """
    兼容新旧checkpoint格式的统一加载函数
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 目标设备
        
    Returns:
        tuple: (model_state_dict, mask_values, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_info = {}
    
    # 检查是否为新格式的完整checkpoint
    if 'model_state_dict' in checkpoint:
        # 新格式：包含完整训练状态
        model_state_dict = checkpoint['model_state_dict']
        mask_values = checkpoint.get('mask_values', list(range(11)))  # 默认11类
        
        # 收集checkpoint信息
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'best_score': checkpoint.get('best_score', 'Unknown'),
            'val_dice': checkpoint.get('val_dice', 'Unknown'),
            'val_miou': checkpoint.get('val_miou', 'Unknown'),
            'seed': checkpoint.get('seed', 'Unknown'),
            'format': 'new'
        }
        
    else:
        # 旧格式：只包含模型权重
        model_state_dict = checkpoint.copy()
        mask_values = model_state_dict.pop('mask_values', list(range(11)))
        checkpoint_info = {'format': 'old'}
    
    return model_state_dict, mask_values, checkpoint_info

def main():
    args = get_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # 检查checkpoint目录结构，支持算子组合子文件夹
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # 检查是否有算子组合的子文件夹
    style_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and '_' in d.name]
    
    # 如果指定了特定的算子组合，只处理该组合
    if args.specific_combo:
        specific_dir = checkpoint_dir / args.specific_combo
        if specific_dir.exists() and specific_dir.is_dir():
            style_dirs = [specific_dir]
            logging.info(f"Processing only specified algorithm combination: {args.specific_combo}")
        else:
            logging.error(f"Specified algorithm combination directory not found: {specific_dir}")
            return
    
    if style_dirs:
        # 如果存在算子组合文件夹，处理每一个
        logging.info(f"Found {len(style_dirs)} algorithm combinations to evaluate:")
        for style_dir in style_dirs:
            logging.info(f"  - {style_dir.name}")
        
        # 用于收集所有结果的列表
        all_results = []
        
        for style_dir in style_dirs:
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing algorithm combination: {style_dir.name}")
            logging.info(f"{'='*60}")
            
            try:
                # 1. 从最后n个checkpoint中找到最佳模型
                logging.info(f"Step 1: Finding best checkpoint from last {args.n_models} models...")
                last_n_checkpoints, style_combo = find_best_checkpoint_from_last_n(
                    style_dir, args.n_models, device
                )
                
                # 2. 评估并选择最佳checkpoint（基于测试集）
                logging.info(f"Step 2: Evaluating checkpoints on test set to find best model...")
                best_checkpoint, best_epoch, best_test_miou, (style1, style2) = evaluate_best_checkpoint(
                    last_n_checkpoints, device, style_combo
                )
                
                # 3. 加载最佳模型
                logging.info(f"Step 3: Loading best model (Epoch {best_epoch}) selected based on test performance...")
                model = UNet(n_channels=3, n_classes=11, style1=style1, style2=style2)
                
                # 使用兼容的checkpoint加载
                model_state_dict, mask_values, checkpoint_info = load_checkpoint_compatible(best_checkpoint, device)
                model.load_state_dict(model_state_dict)
                model.to(device)
                
                logging.info(f'Model loaded from {best_checkpoint}')
                logging.info(f'Best Test mIoU (from selection): {best_test_miou:.4f}')
                
                # 4. 创建测试数据集
                logging.info("Step 4: Loading test dataset for final comprehensive evaluation...")
                test_dataset = BasicDataset(
                    dir_test_img, 
                    dir_test_mask, 
                    scale=args.scale, 
                    normalize=args.normalize, 
                    augment=False
                )
                
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=4, 
                    pin_memory=True
                )
                
                logging.info(f'Test dataset size: {len(test_dataset)} images')
                
                # 5. 在测试集上进行全面评估
                logging.info("Step 5: Running final comprehensive evaluation on test set...")
                results = comprehensive_test_evaluation(model, test_loader, device, args.amp)
                
                # 6. 打印和保存结果（按算子组合分组）
                print_detailed_results(results, CLASS_NAMES)
                save_results_to_csv(results, CLASS_NAMES, args.save_results, style_combo)
                
                # 7. 保存最佳模型信息
                model_info = {
                    'algorithm_combination': style_combo,
                    'style1': style1,
                    'style2': style2,
                    'best_checkpoint': str(best_checkpoint),
                    'best_epoch': best_epoch,
                    'selection_test_miou': best_test_miou,  # 选择时的测试集mIoU
                    'final_test_miou': results['mean_iou'],  # 最终完整评估的测试集mIoU
                    'test_dice': results['mean_dice'],
                    'test_pixel_accuracy': results['overall_pixel_accuracy']
                }
                
                result_path = Path(args.save_results) / style_combo if style_combo else Path(args.save_results)
                pd.DataFrame([model_info]).to_csv(result_path / 'model_info.csv', index=False)
                
                # 收集结果用于最终汇总
                all_results.append(model_info)
                
                print(f"\nEvaluation completed for {style_combo}!")
                print(f"Results saved to: {result_path}")
                print(f"Final Test mIoU: {results['mean_iou']:.4f}")
                print(f"Selection was based on Test mIoU: {best_test_miou:.4f}")
                
            except Exception as e:
                logging.error(f"Failed to process {style_dir.name}: {e}")
                continue
        
        # 8. 保存所有算子组合的汇总结果
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_path = Path(args.save_results) / 'algorithm_comparison_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            
            logging.info(f"\n{'='*60}")
            logging.info("ALGORITHM COMBINATION COMPARISON SUMMARY")
            logging.info(f"{'='*60}")
            
            # 按测试mIoU排序并显示结果
            summary_df_sorted = summary_df.sort_values('final_test_miou', ascending=False)
            print("\nRanking by Final Test mIoU:")
            for idx, row in summary_df_sorted.iterrows():
                print(f"{row['algorithm_combination']:<20} | Final Test mIoU: {row['final_test_miou']:.4f} | Test Dice: {row['test_dice']:.4f}")
            
            print(f"\nDetailed comparison saved to: {summary_path}")
    
    else:
        # 如果没有算子组合文件夹，按原来的方式处理
        logging.info("No algorithm combination folders found, processing default checkpoint directory...")
        
        # 1. 从最后n个checkpoint中找到最佳模型
        logging.info(f"Step 1: Finding best checkpoint from last {args.n_models} models...")
        try:
            last_n_checkpoints, style_combo = find_best_checkpoint_from_last_n(
                args.checkpoint_dir, args.n_models, device
            )
            
            best_checkpoint, best_epoch, best_test_miou, (style1, style2) = evaluate_best_checkpoint(
                last_n_checkpoints, device, style_combo
            )
        except Exception as e:
            logging.error(f"Failed to find best checkpoint: {e}")
            return
        
        # 2. 加载最佳模型
        logging.info(f"Step 2: Loading best model (Epoch {best_epoch})...")
        model = UNet(n_channels=3, n_classes=11, style1=style1, style2=style2)
        
        # 使用兼容的checkpoint加载
        model_state_dict, mask_values, checkpoint_info = load_checkpoint_compatible(best_checkpoint, device)
        model.load_state_dict(model_state_dict)
        model.to(device)
        
        logging.info(f'Model loaded from {best_checkpoint}')
        logging.info(f'Best Test mIoU (from selection): {best_test_miou:.4f}')
        
        # 3. 创建测试数据集
        logging.info("Step 3: Loading test dataset...")
        test_dataset = BasicDataset(
            dir_test_img, 
            dir_test_mask, 
            scale=args.scale, 
            normalize=args.normalize, 
            augment=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        logging.info(f'Test dataset size: {len(test_dataset)} images')
        
        # 4. 在测试集上进行全面评估
        logging.info("Step 4: Running comprehensive evaluation on test set...")
        results = comprehensive_test_evaluation(model, test_loader, device, args.amp)
        
        # 5. 打印和保存结果
        print_detailed_results(results, CLASS_NAMES)
        save_results_to_csv(results, CLASS_NAMES, args.save_results, style_combo)
        
        # 6. 保存最佳模型信息
        model_info = {
            'algorithm_combination': style_combo if style_combo else 'default',
            'style1': style1,
            'style2': style2,
            'best_checkpoint': str(best_checkpoint),
            'best_epoch': best_epoch,
            'selection_test_miou': best_test_miou,  # 选择时的测试集mIoU
            'final_test_miou': results['mean_iou'],  # 最终完整评估的测试集mIoU
            'test_dice': results['mean_dice'],
            'test_pixel_accuracy': results['overall_pixel_accuracy']
        }
        
        result_path = Path(args.save_results) / style_combo if style_combo else Path(args.save_results)
        pd.DataFrame([model_info]).to_csv(result_path / 'model_info.csv', index=False)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {result_path}")
        print(f"Final Test mIoU: {results['mean_iou']:.4f}")
        print(f"Selection was based on Test mIoU: {best_test_miou:.4f}")

if __name__ == '__main__':
    main()
