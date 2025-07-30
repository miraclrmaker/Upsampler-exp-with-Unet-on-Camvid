"""
特征图可视化工具
用于可视化UNet网络中间层的特征图
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """UNet特征提取器，用于提取各层特征图"""
    
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
    def register_hooks(self, layers_to_extract: List[str] = None):
        """注册钩子函数以提取特征图
        
        Args:
            layers_to_extract: 要提取的层名称列表，如果为None则提取所有主要层
        """
        if layers_to_extract is None:
            # 默认提取所有主要层
            layers_to_extract = ['inc', 'down1', 'down2', 'down3', 'down4', 
                                'up1', 'up2', 'up3', 'up4', 'outc']
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    # 对于Up层，输出是tuple，我们取最后一个
                    self.features[name] = output[-1] if len(output) > 1 else output[0]
                else:
                    self.features[name] = output
            return hook
        
        # 清除之前的钩子
        self.clear_hooks()
        
        for name in layers_to_extract:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
        
        logging.info(f"Registered hooks for layers: {layers_to_extract}")
    
    def clear_hooks(self):
        """清除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()
    
    def extract_features(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取特征图
        
        Args:
            input_tensor: 输入张量 [B, C, H, W]
            
        Returns:
            包含各层特征图的字典
        """
        self.features.clear()
        
        # 前向传播以触发钩子
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        return self.features.copy()

def pca_feature_to_rgb(feature_map: np.ndarray, n_components: int = 3) -> np.ndarray:
    """使用PCA将多通道特征图映射到RGB空间
    
    Args:
        feature_map: 特征图 [C, H, W]
        n_components: PCA组件数量（通常为3对应RGB）
        
    Returns:
        RGB图像 [H, W, 3]，值范围[0, 1]
    """
    C, H, W = feature_map.shape
    
    # 记录使用的通道数
    logging.info(f"PCA processing feature map with {C} channels ({H}x{W})")
    
    # 将特征图重塑为 [H*W, C]
    feature_flat = feature_map.transpose(1, 2, 0).reshape(-1, C)  # [H*W, C]
    
    # 处理全零或常数特征
    if np.all(feature_flat == 0) or np.std(feature_flat) < 1e-8:
        logging.warning("Feature map contains all zeros or constant values, returning black image")
        return np.zeros((H, W, 3))
    
    # 标准化特征
    scaler = StandardScaler()
    try:
        feature_normalized = scaler.fit_transform(feature_flat)
    except:
        # 如果标准化失败，使用简单归一化
        feature_normalized = feature_flat
        feature_std = np.std(feature_normalized, axis=0, keepdims=True)
        feature_std[feature_std == 0] = 1  # 避免除零
        feature_normalized = (feature_normalized - np.mean(feature_normalized, axis=0, keepdims=True)) / feature_std
    
    # 应用PCA
    n_components = min(n_components, C, feature_normalized.shape[0])  # 确保组件数不超过可用维度
    if n_components < 3:
        # 如果组件数不足3，用零填充
        pca_result = np.zeros((feature_normalized.shape[0], 3))
        if n_components > 0:
            pca = PCA(n_components=n_components)
            pca_components = pca.fit_transform(feature_normalized)
            pca_result[:, :n_components] = pca_components
        
        logging.info(f"PCA with {n_components} components (padded to 3 for RGB)")
    else:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(feature_normalized)
        
        # 记录PCA信息
        explained_variance = pca.explained_variance_ratio_
        logging.info(f"PCA from {C} channels: explained variance R={explained_variance[0]:.3f}, "
                    f"G={explained_variance[1]:.3f}, B={explained_variance[2]:.3f}, "
                    f"total={sum(explained_variance):.3f}")
    
    # 将PCA结果重塑回图像形状
    pca_image = pca_result.reshape(H, W, 3)
    
    # 归一化到[0, 1]范围
    for i in range(3):
        channel = pca_image[:, :, i]
        if np.max(channel) > np.min(channel):
            pca_image[:, :, i] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
        else:
            pca_image[:, :, i] = 0.5  # 如果通道是常数，设为中间值
    
    return pca_image

def visualize_feature_maps_pca(features: Dict[str, torch.Tensor], 
                              save_dir: str,
                              image_name: str,
                              show_individual_channels: bool = True,
                              max_individual_channels: int = None,
                              figsize_per_item: Tuple[int, int] = (4, 4)) -> None:
    """使用PCA可视化特征图，同时显示PCA合成图和个别通道
    
    Args:
        features: 特征图字典
        save_dir: 保存目录
        image_name: 图像名称
        show_individual_channels: 是否同时显示个别通道
        max_individual_channels: 最大显示的个别通道数（None表示显示所有通道）
        figsize_per_item: 每个子图的大小
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for layer_name, feature_tensor in features.items():
        # 转换为numpy数组 [B, C, H, W] -> [C, H, W]
        if len(feature_tensor.shape) == 4:
            feature_np = feature_tensor[0].cpu().numpy()  # 取第一个batch
        else:
            feature_np = feature_tensor.cpu().numpy()
        
        C, H, W = feature_np.shape
        
        # 生成PCA RGB图像
        pca_rgb = pca_feature_to_rgb(feature_np)
        
        if show_individual_channels and C > 3:
            # 决定显示多少个个别通道
            if max_individual_channels is None:
                num_individual = C  # 显示所有通道
                logging.info(f"Showing ALL {C} channels for layer {layer_name}")
            else:
                num_individual = min(max_individual_channels, C)
                logging.info(f"Showing {num_individual}/{C} channels for layer {layer_name}")
            
            total_plots = 1 + num_individual  # PCA图 + 个别通道
            
            # 计算布局 - 对于大量通道，增加列数
            max_cols = 8 if num_individual > 32 else 5
            cols = min(max_cols, total_plots)
            rows = (total_plots + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, 
                                    figsize=(cols * figsize_per_item[0], 
                                            rows * figsize_per_item[1]))
            
            if total_plots == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten() if cols > 1 else [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'{layer_name} - PCA Visualization\nShape: {feature_np.shape}', 
                        fontsize=16, y=0.98)
            
            # 显示PCA RGB图
            axes[0].imshow(pca_rgb)
            axes[0].set_title(f'PCA RGB Composite\n({C} channels → RGB)', fontsize=10, weight='bold')
            axes[0].axis('off')
            
            # 显示个别通道
            for i in range(num_individual):
                ax_idx = i + 1
                if ax_idx < len(axes):
                    channel = feature_np[i]
                    normalized_channel = normalize_feature_map(channel)
                    
                    im = axes[ax_idx].imshow(normalized_channel, cmap='viridis')
                    axes[ax_idx].set_title(f'Ch {i}\nMin: {channel.min():.3f}\nMax: {channel.max():.3f}', 
                                          fontsize=8)
                    axes[ax_idx].axis('off')
            
            # 隐藏多余的子图
            for i in range(total_plots, len(axes)):
                axes[i].axis('off')
        
        else:
            # 只显示PCA图
            fig, ax = plt.subplots(1, 1, figsize=figsize_per_item)
            ax.imshow(pca_rgb)
            ax.set_title(f'{layer_name} - PCA RGB Composite\nShape: {feature_np.shape} → RGB', 
                        fontsize=14, weight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = save_dir / f'{image_name}_{layer_name}_pca_features.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logging.info(f'PCA feature maps for {layer_name} saved to {save_path}')

def create_pca_comparison_grid(features: Dict[str, torch.Tensor],
                              save_dir: str,
                              image_name: str,
                              grid_size: Tuple[int, int] = None) -> None:
    """创建所有层的PCA特征图对比网格
    
    Args:
        features: 特征图字典
        save_dir: 保存目录
        image_name: 图像名称
        grid_size: 网格大小 (rows, cols)，如果为None则自动计算
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成所有层的PCA图像
    pca_images = {}
    layer_info = {}
    
    for layer_name, feature_tensor in features.items():
        if len(feature_tensor.shape) == 4:
            feature_np = feature_tensor[0].cpu().numpy()
        else:
            feature_np = feature_tensor.cpu().numpy()
        
        pca_rgb = pca_feature_to_rgb(feature_np)
        pca_images[layer_name] = pca_rgb
        layer_info[layer_name] = {
            'shape': feature_np.shape,
            'channels': feature_np.shape[0],
            'spatial': feature_np.shape[1:]
        }
    
    # 计算网格布局
    num_layers = len(pca_images)
    if grid_size is None:
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
    else:
        rows, cols = grid_size
    
    # 创建网格图
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(f'PCA Feature Maps Comparison - {image_name}', fontsize=18, y=0.98)
    
    if num_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten() if cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # 显示每层的PCA图
    for i, (layer_name, pca_image) in enumerate(pca_images.items()):
        if i < len(axes):
            axes[i].imshow(pca_image)
            info = layer_info[layer_name]
            axes[i].set_title(f'{layer_name}\n{info["channels"]}ch, {info["spatial"]}', 
                             fontsize=10, weight='bold')
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 保存对比图
    save_path = save_dir / f'{image_name}_pca_comparison_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f'PCA comparison grid saved to {save_path}')

def normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    """归一化特征图到[0, 1]范围"""
    fmin = feature_map.min()
    fmax = feature_map.max()
    if fmax > fmin:
        return (feature_map - fmin) / (fmax - fmin)
    else:
        return np.zeros_like(feature_map)

def create_feature_summary(features: Dict[str, torch.Tensor], 
                          save_dir: str,
                          image_name: str) -> None:
    """创建特征图统计摘要
    
    Args:
        features: 特征图字典
        save_dir: 保存目录
        image_name: 图像名称
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建摘要图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Feature Map Summary - {image_name}', fontsize=16)
    
    layer_names = list(features.keys())
    layer_shapes = []
    layer_means = []
    layer_stds = []
    layer_activations = []
    
    # 收集统计信息
    for layer_name, feature_tensor in features.items():
        if len(feature_tensor.shape) == 4:
            feature_np = feature_tensor[0].cpu().numpy()  # 取第一个batch
        else:
            feature_np = feature_tensor.cpu().numpy()
        
        layer_shapes.append(f'{feature_np.shape}')
        layer_means.append(feature_np.mean())
        layer_stds.append(feature_np.std())
        
        # 计算激活强度（平均绝对值）
        activation_strength = np.abs(feature_np).mean()
        layer_activations.append(activation_strength)
    
    # 绘制统计图表
    x_pos = np.arange(len(layer_names))
    
    # 1. 激活强度
    axes[0, 0].bar(x_pos, layer_activations, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Average Activation Strength')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Activation Strength')
    
    # 2. 均值
    axes[0, 1].plot(x_pos, layer_means, 'o-', color='green', linewidth=2, markersize=6)
    axes[0, 1].set_title('Feature Map Means')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 标准差
    axes[1, 0].plot(x_pos, layer_stds, 's-', color='red', linewidth=2, markersize=6)
    axes[1, 0].set_title('Feature Map Standard Deviations')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 形状信息表格
    axes[1, 1].axis('off')
    table_data = []
    for i, (name, shape, mean, std, activation) in enumerate(
        zip(layer_names, layer_shapes, layer_means, layer_stds, layer_activations)):
        table_data.append([name, shape, f'{mean:.3f}', f'{std:.3f}', f'{activation:.3f}'])
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Layer', 'Shape', 'Mean', 'Std', 'Activation'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    axes[1, 1].set_title('Layer Statistics', pad=20)
    
    plt.tight_layout()
    
    # 保存摘要
    save_path = save_dir / f'{image_name}_feature_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f'Feature summary saved to {save_path}')

def visualize_attention_maps(features: Dict[str, torch.Tensor],
                           save_dir: str,
                           image_name: str,
                           original_size: Tuple[int, int] = None) -> None:
    """创建注意力图可视化（基于特征图的空间激活强度）
    
    Args:
        features: 特征图字典
        save_dir: 保存目录
        image_name: 图像名称
        original_size: 原图尺寸 (H, W)，用于上采样注意力图
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    attention_maps = {}
    
    # 为每层计算注意力图
    for layer_name, feature_tensor in features.items():
        if len(feature_tensor.shape) == 4:
            feature_np = feature_tensor[0].cpu().numpy()  # [C, H, W]
        else:
            continue
        
        # 计算每个空间位置的激活强度（所有通道的平均）
        attention_map = np.mean(np.abs(feature_np), axis=0)  # [H, W]
        attention_maps[layer_name] = attention_map
    
    # 创建注意力图可视化
    num_layers = len(attention_maps)
    cols = min(3, num_layers)
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle(f'Attention Maps - {image_name}', fontsize=16)
    
    if num_layers == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        axes = axes.flatten()
    elif rows > 1:
        axes = axes.flatten()
    
    for i, (layer_name, attention_map) in enumerate(attention_maps.items()):
        ax = axes[i] if num_layers > 1 else axes[0]
        
        # 归一化注意力图
        normalized_map = normalize_feature_map(attention_map)
        
        # 如果提供了原图尺寸，上采样注意力图
        if original_size is not None:
            from scipy.ndimage import zoom
            scale_h = original_size[0] / normalized_map.shape[0]
            scale_w = original_size[1] / normalized_map.shape[1]
            normalized_map = zoom(normalized_map, (scale_h, scale_w), order=1)
        
        # 显示注意力图
        im = ax.imshow(normalized_map, cmap='jet', alpha=0.8)
        ax.set_title(f'{layer_name}\nShape: {attention_map.shape}', fontsize=10)
        ax.axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 保存注意力图
    save_path = save_dir / f'{image_name}_attention_maps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f'Attention maps saved to {save_path}')

def create_comprehensive_visualization(model,
                                     input_tensor: torch.Tensor,
                                     save_dir: str,
                                     image_name: str,
                                     original_size: Tuple[int, int] = None,
                                     layers_to_extract: List[str] = None,
                                     max_channels: int = None,
                                     use_pca: bool = True,
                                     show_individual_channels: bool = True) -> Dict[str, torch.Tensor]:
    """创建全面的特征可视化
    
    Args:
        model: UNet模型
        input_tensor: 输入张量
        save_dir: 保存目录
        image_name: 图像名称
        original_size: 原图尺寸
        layers_to_extract: 要提取的层
        max_channels: 最大显示通道数（仅在show_individual_channels=True时使用，None表示显示所有通道）
        use_pca: 是否使用PCA方法进行特征图可视化
        show_individual_channels: 是否同时显示个别通道（仅在use_pca=True时有效）
        
    Returns:
        提取的特征图字典
    """
    # 创建特征提取器
    extractor = FeatureExtractor(model)
    extractor.register_hooks(layers_to_extract)
    
    try:
        # 提取特征图
        features = extractor.extract_features(input_tensor)
        
        # 根据选择的方法创建特征图可视化
        if use_pca:
            logging.info("Using PCA-based feature map visualization for ALL channels")
            # 使用PCA方法可视化特征图（处理所有通道）
            visualize_feature_maps_pca(
                features, save_dir, image_name, 
                show_individual_channels=show_individual_channels,
                max_individual_channels=max_channels  # None表示显示所有通道
            )
            # 创建PCA对比网格
            create_pca_comparison_grid(features, save_dir, image_name)
        else:
            logging.info("Using traditional channel-wise feature map visualization")
            # 使用传统方法显示个别通道
            max_channels_to_use = max_channels if max_channels is not None else 16
            visualize_feature_maps_traditional(features, save_dir, image_name, max_channels_to_use)
        
        # 创建特征统计摘要和注意力图（这些保持不变）
        create_feature_summary(features, save_dir, image_name)
        visualize_attention_maps(features, save_dir, image_name, original_size)
        
        logging.info(f'Comprehensive visualization completed for {image_name}')
        
        return features
        
    finally:
        # 清理钩子
        extractor.clear_hooks()

def visualize_feature_maps_traditional(features: Dict[str, torch.Tensor], 
                                      save_dir: str,
                                      image_name: str,
                                      max_channels: int = 16,
                                      figsize_per_channel: Tuple[int, int] = (2, 2)) -> None:
    """传统的特征图可视化方法（显示个别通道）
    
    Args:
        features: 特征图字典
        save_dir: 保存目录
        image_name: 图像名称
        max_channels: 每层最大显示的通道数
        figsize_per_channel: 每个通道子图的大小
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for layer_name, feature_tensor in features.items():
        # 转换为numpy数组 [B, C, H, W] -> [C, H, W]
        if len(feature_tensor.shape) == 4:
            feature_np = feature_tensor[0].cpu().numpy()  # 取第一个batch
        else:
            feature_np = feature_tensor.cpu().numpy()
        
        num_channels = min(feature_np.shape[0], max_channels)
        
        # 计算子图布局
        cols = min(4, num_channels)
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, 
                                figsize=(cols * figsize_per_channel[0], 
                                        rows * figsize_per_channel[1]))
        
        if num_channels == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = axes.flatten()
        elif rows > 1:
            axes = axes.flatten()
        
        fig.suptitle(f'{layer_name} - Shape: {feature_np.shape}', fontsize=14, y=0.98)
        
        for i in range(num_channels):
            ax = axes[i] if num_channels > 1 else axes[0]
            
            # 归一化特征图
            feature_map = normalize_feature_map(feature_np[i])
            
            # 显示特征图
            im = ax.imshow(feature_map, cmap='viridis', aspect='auto')
            ax.set_title(f'Channel {i}\nMin: {feature_np[i].min():.3f}\nMax: {feature_np[i].max():.3f}', 
                        fontsize=8)
            ax.axis('off')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = save_dir / f'{image_name}_{layer_name}_traditional_features.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logging.info(f'Traditional feature maps for {layer_name} saved to {save_path}')
