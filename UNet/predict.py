import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 设置后端为不需要GUI的模式
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def create_simple_pca_visualization(model, input_tensor, save_dir, image_name):
    """创建只包含PCA合成图的简化特征可视化，所有层合并在一张图中"""
    import os
    from sklearn.decomposition import PCA
    import math
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取特征的钩子函数
    features = {}
    hooks = []
    
    def get_features(name):
        def hook(model, input, output):
            if isinstance(output, (list, tuple)):
                output = output[0]  # 取第一个输出
            features[name] = output.detach()
        return hook
    
    # 直接使用UNet的属性来注册钩子
    layer_configs = [
        ('inc', model.inc),
        ('down1', model.down1),
        ('down2', model.down2),
        ('down3', model.down3),
        ('down4', model.down4),
        ('up1', model.up1),
        ('up2', model.up2),
        ('up3', model.up3),
        ('up4', model.up4),
        ('outc', model.outc)
    ]
    
    for name, layer in layer_configs:
        hooks.append(layer.register_forward_hook(get_features(name)))
    
    logging.info(f"Registered hooks for {len(layer_configs)} layers")
    
    # 前向传播提取特征
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 清理钩子
    for hook in hooks:
        hook.remove()
    
    logging.info(f"Extracted features from {len(features)} layers")
    
    # 收集所有有效的PCA图像
    pca_images = {}
    layer_info = {}
    
    for layer_name, feature_map in features.items():
        if feature_map.dim() != 4:  # 确保是4D张量 [B, C, H, W]
            logging.warning(f"Unexpected feature shape for {layer_name}: {feature_map.shape}")
            continue
            
        feature_map = feature_map[0]  # 取第一个batch [C, H, W]
        C, H, W = feature_map.shape
        
        logging.info(f"Processing {layer_name}: shape={feature_map.shape}")
        
        if C < 3:  # 如果通道数小于3，跳过PCA
            logging.warning(f"Skipping {layer_name}: insufficient channels ({C})")
            continue
            
        # 重塑为 [H*W, C] 进行PCA
        feature_flat = feature_map.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
        
        # 应用PCA到RGB空间
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(feature_flat)
        
        # 归一化到0-1范围
        pca_min = pca_result.min(axis=0)
        pca_max = pca_result.max(axis=0)
        pca_normalized = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)
        
        # 重塑回图像格式
        pca_image = pca_normalized.reshape(H, W, 3)
        
        # 保存到字典中
        pca_images[layer_name] = pca_image
        layer_info[layer_name] = {
            'shape': feature_map.shape,
            'channels': C,
            'explained_variance': pca.explained_variance_ratio_.sum()
        }
    
    # 创建合并的可视化图
    if len(pca_images) > 0:
        # 计算最优子图布局，避免空白子图
        num_layers = len(pca_images)
        if num_layers <= 5:
            rows, cols = 1, num_layers
        elif num_layers <= 10:
            rows, cols = 2, 5
        else:
            cols = 5
            rows = math.ceil(num_layers / cols)
        
        # 创建紧凑版本的图，增大尺寸给标题留空间
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        fig.suptitle(f'PCA Feature Visualization - {image_name}', fontsize=16, y=0.98)
        
        # 处理单行或单列的情况
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 显示每个层的PCA图
        layer_names = list(pca_images.keys())
        for i, layer_name in enumerate(layer_names):
            if rows == 1:
                ax = axes[0, i] if cols > 1 else axes[i]
            else:
                row = i // cols
                col = i % cols
                ax = axes[row, col]
            
            pca_image = pca_images[layer_name]
            info = layer_info[layer_name]
            
            ax.imshow(pca_image)
            ax.set_title(f'{layer_name}\n{info["channels"]}ch', fontsize=11, weight='bold', pad=10)
            ax.axis('off')
        
        # 移除多余的子图（而不是隐藏）
        if rows > 1 or cols > 1:
            for i in range(num_layers, rows * cols):
                row = i // cols
                col = i % cols
                if rows == 1:
                    fig.delaxes(axes[0, col] if cols > 1 else axes[col])
                else:
                    fig.delaxes(axes[row, col])
        
        plt.tight_layout(pad=2.0, h_pad=4.0)  # 增加padding避免文字重叠，h_pad增大行间距
        
        # 保存紧凑版本
        save_path = os.path.join(save_dir, f'{image_name}_all_layers_pca.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f'PCA visualization saved: {save_path}')
    
    else:
        logging.warning("No valid PCA images generated")

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5, extract_features=False):
    net.eval()
    
    # 修复：使用和训练时完全相同的预处理参数
    img = torch.from_numpy(BasicDataset.preprocess(
        mask_values=None,
        pil_img=full_img, 
        scale=scale_factor, 
        is_mask=False,
        normalize=True,
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]    # ImageNet标准差
    ))
    
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # 常规预测
        output = net(img)
        features = None
            
        # 确保输出尺寸正确
        if output.shape[-2:] != (full_img.size[1], full_img.size[0]):
            output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear', align_corners=False)
        
        if net.n_classes > 1:
            # 多类分割使用softmax + argmax
            mask = F.softmax(output, dim=1).argmax(dim=1)
        else:
            # 二分类使用sigmoid
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].cpu().numpy().astype(np.uint8), features, img

def mask_to_color(mask, colormap):
    """Convert a mask with class indices to an RGB image using the given colormap."""
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    
    # 创建RGB输出
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 处理有效类别（0到len(colormap)-1）
    for i, color in enumerate(colormap):
        class_pixels = (mask == i)
        color_mask[class_pixels] = color
    
    # 处理unlabelled像素（255） - 显示为黑色
    unlabelled_pixels = (mask == 255)
    color_mask[unlabelled_pixels] = [0, 0, 0]  # 黑色表示unlabelled
    
    return color_mask.astype(np.uint8)

def get_camvid_colormap():
    """CamVid数据集的标准颜色映射"""
    return np.array([
        [128, 128, 128],  # Sky
        [128, 0, 0],      # Building
        [192, 192, 128],  # Pole
        [128, 64, 128],   # Road
        [0, 0, 192],      # Sidewalk
        [128, 128, 0],    # Tree
        [192, 128, 128],  # SignSymbol
        [64, 64, 128],    # Fence
        [64, 0, 128],     # Car
        [64, 64, 0],      # Pedestrian
        [0, 128, 192],    # Bicyclist
    ], dtype=np.uint8)

def rgb_to_class(label_arr, colormap):
    """改进的RGB到类别索引转换，正确处理unlabelled像素"""
    if label_arr.ndim == 2:
        return label_arr  # 已经是类别索引
    
    h, w, c = label_arr.shape
    if c == 1:
        return label_arr.squeeze()
    
    label_flat = label_arr.reshape(-1, 3)
    colormap = np.array(colormap, dtype=np.uint8)
    
    # 初始化为255（unlabelled）
    class_idx = np.full((label_flat.shape[0],), 255, dtype=np.uint8)
    
    # 第一遍：精确匹配
    for i, color in enumerate(colormap):
        exact_matches = np.all(label_flat == color, axis=1)
        class_idx[exact_matches] = i
    
    # 第二遍：对剩余未匹配的像素进行容差匹配
    unmatched_mask = (class_idx == 255)
    if unmatched_mask.sum() > 0:
        unmatched_pixels = label_flat[unmatched_mask]
        
        for i, color in enumerate(colormap):
            if unmatched_mask.sum() == 0:
                break
                
            # 计算未匹配像素与当前颜色的距离
            distances = np.sqrt(np.sum((unmatched_pixels.astype(np.float32) - color.astype(np.float32))**2, axis=1))
            close_matches = distances <= 5.0  # 允许一定的颜色差异
            
            if close_matches.sum() > 0:
                # 获取原始索引并更新
                unmatched_indices = np.where(unmatched_mask)[0]
                class_idx[unmatched_indices[close_matches]] = i
                
                # 更新未匹配mask
                unmatched_mask[unmatched_indices[close_matches]] = False
                unmatched_pixels = unmatched_pixels[~close_matches]
    
    # 最后处理：确保真正的黑色像素（unlabelled）保持为255
    # 检查colormap中是否有黑色
    has_black_class = np.any(np.all(colormap == [0, 0, 0], axis=1))
    
    if not has_black_class:
        # 如果colormap中没有黑色，则所有黑色像素都是unlabelled
        black_pixels = np.all(label_flat == [0, 0, 0], axis=1)
        class_idx[black_pixels] = 255
    
    # 调试信息
    mapped_pixels = (class_idx != 255).sum()
    unlabelled_pixels = (class_idx == 255).sum()
    total_pixels = len(class_idx)
    
    logging.info(f"RGB to class mapping: {mapped_pixels}/{total_pixels} ({100*mapped_pixels/total_pixels:.1f}%) mapped, "
                f"{unlabelled_pixels} ({100*unlabelled_pixels/total_pixels:.1f}%) unlabelled")
    
    # 显示类别分布
    for i in range(len(colormap)):
        count = (class_idx == i).sum()
        if count > 0:
            logging.info(f"  Class {i}: {count} pixels ({100*count/total_pixels:.2f}%)")
    
    # 警告：如果有太多未映射的像素
    if unlabelled_pixels > total_pixels * 0.15:  # 超过15%未映射
        unmapped_mask = (class_idx == 255)
        unmapped_colors = np.unique(label_flat[unmapped_mask], axis=0)
        logging.warning(f"Large number of unmapped pixels! Unique unmapped colors (first 5): {unmapped_colors[:5].tolist()}")
    
    return class_idx.reshape(h, w)

def compute_miou_and_acc(pred, gt, num_classes, ignore_index=255):
    """计算mIoU和像素精度，忽略unlabelled像素"""
    # 确保数据类型一致
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    
    # 限制类别范围
    pred = np.clip(pred, 0, max(num_classes - 1, 255))
    
    ious = []
    valid_classes = 0
    
    # 创建有效像素mask
    valid_mask = (gt != ignore_index)
    
    for cls in range(num_classes):
        # 只在有效像素上计算
        pred_inds = (pred == cls) & valid_mask
        gt_inds = (gt == cls) & valid_mask
        
        intersection = (pred_inds & gt_inds).sum()
        union = (pred_inds | gt_inds).sum()
        
        if union == 0:
            # 如果该类别在有效像素中不存在，跳过
            continue
        else:
            iou = intersection / union
            ious.append(iou)
            valid_classes += 1
    
    # 计算像素精度（只考虑有效像素）
    if valid_mask.sum() > 0:
        pixel_acc = ((pred == gt) & valid_mask).sum() / valid_mask.sum()
    else:
        pixel_acc = 0.0
    
    miou = np.mean(ious) if len(ious) > 0 else 0.0
    
    return miou, pixel_acc, ious, valid_classes

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
        logging.info('Detected new checkpoint format with full training state')
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
        logging.info('Detected old checkpoint format (model weights only)')
        model_state_dict = checkpoint.copy()
        mask_values = model_state_dict.pop('mask_values', list(range(11)))
        checkpoint_info = {'format': 'old'}
    
    return model_state_dict, mask_values, checkpoint_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/root/1/checkpoints/max-pooling_dysample_lp-dynamic/checkpoint_epoch113.pth', 
                       metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', 
                       help='Filenames of input images', required=True)
    parser.add_argument('--output-dir', '-o', default='./predictions/', 
                       help='Output directory for predictions (will create algorithm-specific subdirectories)')
    parser.add_argument('--classes', '-c', type=int, default=11, help='Number of classes')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for input images')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed analysis')
    parser.add_argument('--visualize-features', action='store_true', 
                       help='Enable PCA feature map visualization (saves to features/ subdirectory)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 从模型文件名推断算子组合（如果可能）
    model_path = args.model
    style1, style2 = 'max-pooling', 'bilinear'  # 默认值
    
    # 尝试从路径推断算子组合
    if 'checkpoints/' in model_path:
        path_parts = model_path.split('/')
        for part in path_parts:
            if '_' in part and not part.startswith('checkpoint'):
                # 可能是算子组合文件夹名
                style_parts = part.split('_')
                if len(style_parts) >= 2:
                    style1 = style_parts[0]
                    style2 = '_'.join(style_parts[1:])  # 支持多段式的算子名称如dysample_lp
                    logging.info(f'Detected algorithm combination from path: {style1}_{style2}')
                    break
    
    # 创建按算子组合分类的输出目录
    algorithm_dir = f"{style1}_{style2}"
    full_output_dir = os.path.join(args.output_dir, algorithm_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    logging.info(f'Results will be saved to algorithm-specific directory: {full_output_dir}')
    
    # 如果启用特征可视化，创建特征目录
    if args.visualize_features:
        feature_dir = os.path.join(full_output_dir, 'features')
        os.makedirs(feature_dir, exist_ok=True)
        
        logging.info(f'Feature visualization enabled. PCA features will be saved to: {feature_dir}')
        logging.info('Only PCA composite visualizations will be generated for each layer.')
    
    # 加载模型
    net = UNet(n_channels=3, n_classes=args.classes, style1=style1, style2=style2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    logging.info(f'Model configuration: style1={style1}, style2={style2}')
    
    net.to(device=device)
    
    # 兼容新旧checkpoint格式的加载逻辑
    logging.info(f'Loading checkpoint {args.model}')
    model_state_dict, mask_values, checkpoint_info = load_checkpoint_compatible(args.model, device)
    
    # 加载模型权重
    net.load_state_dict(model_state_dict)
    logging.info('Model loaded successfully!')
    
    # 获取颜色映射
    colormap = get_camvid_colormap()
    
    for filename in args.input:
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        
        # 预测
        mask, features, input_tensor = predict_img(
            net=net, 
            full_img=img, 
            device=device, 
            scale_factor=args.scale,
            extract_features=False  # 不再需要这个参数
        )
        
        # 转换为彩色图像
        mask_rgb = mask_to_color(mask, colormap)
        
        # 准备文件名
        img_name = os.path.basename(filename)
        base_name = os.path.splitext(img_name)[0]
        
        # 特征可视化（如果启用）
        if args.visualize_features:
            logging.info(f'Generating PCA feature visualizations for {img_name}...')
            
            try:
                # 创建该图像的特征子目录
                img_feature_dir = os.path.join(feature_dir, base_name)
                
                # 创建简化的PCA特征可视化
                create_simple_pca_visualization(
                    model=net,
                    input_tensor=input_tensor,
                    save_dir=img_feature_dir,
                    image_name=base_name
                )
                
                logging.info(f'PCA feature visualizations saved to {img_feature_dir}')
                
            except Exception as e:
                logging.error(f'Failed to generate feature visualizations: {e}')
        
        # 查找对应的标签文件
        img_dir = os.path.dirname(filename)
        label_dir = img_dir.replace('train', 'train_labels').replace('test', 'test_labels')
        label_name = img_name.replace('.png', '_L.png')
        label_path = os.path.join(label_dir, label_name)
        
        # 创建可视化
        if os.path.exists(label_path):
            # 加载真实标签
            label_img = Image.open(label_path)
            label_arr = np.array(label_img)
            
            # 调试信息：显示原始标签的统计
            if args.debug:
                logging.info(f"Original label shape: {label_arr.shape}")
                if label_arr.ndim == 3:
                    unique_colors = np.unique(label_arr.reshape(-1, 3), axis=0)
                    logging.info(f"Unique colors in original label: {len(unique_colors)}")
                    logging.info(f"First 10 unique colors: {unique_colors[:10].tolist()}")
                    
                    # 检查黑色像素
                    black_count = np.all(label_arr == [0, 0, 0], axis=2).sum()
                    total_pixels = label_arr.shape[0] * label_arr.shape[1]
                    logging.info(f"Black pixels [0,0,0]: {black_count}/{total_pixels} ({100*black_count/total_pixels:.2f}%)")
            
            if label_arr.ndim == 3:
                label_arr = rgb_to_class(label_arr, colormap)
            
            # 调试信息：显示转换后的类别分布
            if args.debug:
                unique_classes = np.unique(label_arr)
                logging.info(f"Classes after conversion: {unique_classes}")
                for cls in unique_classes:
                    count = (label_arr == cls).sum()
                    total = label_arr.size
                    if cls == 255:
                        logging.info(f"  Unlabelled (255): {count}/{total} ({100*count/total:.2f}%)")
                    else:
                        logging.info(f"  Class {cls}: {count}/{total} ({100*count/total:.2f}%)")
            
            label_rgb = mask_to_color(label_arr, colormap)
            
            # 计算指标 - 忽略unlabelled像素
            miou, acc, ious, valid_classes = compute_miou_and_acc(mask, label_arr, args.classes, ignore_index=255)
            
            logging.info(f"Image: {img_name}")
            logging.info(f"  mIoU (ignoring unlabelled): {miou:.4f}")
            logging.info(f"  Pixel Accuracy (ignoring unlabelled): {acc:.4f}")
            logging.info(f"  Valid classes: {valid_classes}")
            
            # 添加调试信息
            unique_pred = np.unique(mask)
            unique_gt = np.unique(label_arr)
            logging.info(f"  Prediction unique values: {unique_pred}")
            logging.info(f"  Ground truth unique values: {unique_gt}")
            logging.info(f"  Unlabelled pixels in GT: {(label_arr == 255).sum()}")
            
            # 创建三列可视化
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].imshow(np.asarray(img))
            axs[0].set_title('Input Image')
            axs[0].axis('off')
            
            axs[1].imshow(label_rgb)
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')
            
            axs[2].imshow(mask_rgb)
            axs[2].set_title(f'Prediction\nmIoU: {miou:.4f}, Acc: {acc:.4f}')
            axs[2].axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(full_output_dir, f'{base_name}_comparison.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f'Comparison saved to {output_path}')
            
        else:
            # 只有预测结果
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            axs[0].imshow(np.asarray(img))
            axs[0].set_title('Input Image')
            axs[0].axis('off')
            
            axs[1].imshow(mask_rgb)
            axs[1].set_title('Predicted Mask')
            axs[1].axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(full_output_dir, f'{base_name}_prediction.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f'Prediction saved to {output_path}')
        
        # 保存原始预测mask
        mask_path = os.path.join(full_output_dir, f'{base_name}_mask.png')
        Image.fromarray(mask_rgb).save(mask_path)
        logging.info(f'Mask saved to {mask_path}')

    logging.info(f'All predictions completed! Results saved in {full_output_dir}')