import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import random


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    # print(idx + mask_suffix +'_L' + '.*')
    mask_file = list(mask_dir.glob(idx + mask_suffix +'_L' + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', 
                 normalize: bool = False, return_raw: bool = False, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.normalize = normalize
        self.return_raw = return_raw  # 是否返回原始PIL图像（用于可视化）
        self.augment = augment  # 是否启用数据增强
        
        # 数据增强参数
        self.scale_factors = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # 按用户要求的缩放范围
        
        # 用户指定的裁剪尺寸
        self.crop_size = (360, 480)
        
        # ImageNet预训练模型的标准归一化参数
        # 如果模型使用ImageNet预训练权重，建议使用这些值
        self.mean = [0.485, 0.456, 0.406]  # ImageNet均值
        self.std = [0.229, 0.224, 0.225]   # ImageNet标准差
        
        # CamVid数据集的统计信息（可选，基于实际数据计算）
        # 如果想使用数据集特定的归一化，可以替换上面的值

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Loading class definition from class_dict.csv')
        
        # 加载标准的CamVid类别定义，但排除unlabelled类
        import pandas as pd
        import os
        class_dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CamVid', 'class_dict.csv')
        if os.path.exists(class_dict_path):
            class_dict = pd.read_csv(class_dict_path)
            # 排除unlabelled类，只保留前11个有意义的类别
            class_dict_valid = class_dict[class_dict['name'] != 'unlabelled'].reset_index(drop=True)
            self.mask_values = class_dict_valid[['r', 'g', 'b']].values.tolist()
            self.unlabelled_color = [0, 0, 0]  # 存储unlabelled的颜色用于映射
            logging.info(f'Using predefined class colors from {class_dict_path}')
            logging.info(f'Excluded unlabelled class, using {len(self.mask_values)} valid classes')
        else:
            # 备用方案：扫描文件获取unique values
            logging.info('class_dict.csv not found, scanning mask files to determine unique values')
            with Pool() as p:
                unique = list(tqdm(
                    p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                    total=len(self.ids)
                ))
            all_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
            # 排除黑色（unlabelled）
            self.mask_values = [v for v in all_values if v != [0, 0, 0]]
            self.unlabelled_color = [0, 0, 0]
        logging.info(f'Valid mask values (excluding unlabelled): {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    def apply_augmentation(self, img, mask):
        """改进的数据增强：智能缩放、随机镜像、自适应裁剪"""
        if not self.augment:
            return img, mask
            
        w, h = img.size
        crop_h, crop_w = self.crop_size
        
        # 1. 智能随机缩放
        # 计算最小缩放因子，确保缩放后仍能进行有意义的裁剪
        min_scale_w = crop_w / w
        min_scale_h = crop_h / h
        min_scale = max(min_scale_w, min_scale_h) * 1.1  # 增加10%余量
        
        # 过滤出合理的缩放因子
        valid_scales = [s for s in self.scale_factors if s >= min_scale]
        if not valid_scales:
            valid_scales = [min_scale]
        
        scale_factor = random.choice(valid_scales)
        
        # 应用缩放
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        img = img.resize((new_w, new_h), Image.BICUBIC)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # 2. 随机水平翻转
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 3. 智能裁剪或填充
        w, h = img.size
        
        if w >= crop_w and h >= crop_h:
            # 随机裁剪
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            
            img = img.crop((left, top, left + crop_w, top + crop_h))
            mask = mask.crop((left, top, left + crop_w, top + crop_h))
        else:
            # 需要填充 - 使用统一的策略
            new_img = Image.new('RGB', (crop_w, crop_h), (0, 0, 0))
            
            # 统一使用黑色填充，在预处理时会被正确处理为unlabelled
            if mask.mode == 'L':
                new_mask = Image.new('L', (crop_w, crop_h), 0)  # 统一使用0
            else:
                new_mask = Image.new('RGB', (crop_w, crop_h), (0, 0, 0))
            
            # 居中粘贴
            paste_x = (crop_w - w) // 2
            paste_y = (crop_h - h) // 2
            
            new_img.paste(img, (paste_x, paste_y))
            new_mask.paste(mask, (paste_x, paste_y))
            
            img = new_img
            mask = new_mask
        
        return img, mask

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, normalize=True, mean=None, std=None):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # 初始化为255（ignore_index），这样unlabelled像素会被忽略
            mask = np.full((newH, newW), 255, dtype=np.int64)
            
            # 处理不同格式的标签图像
            if img.ndim == 2:
                # 灰度标签图像：直接映射像素值到类别
                for i, v in enumerate(mask_values):
                    if isinstance(v, list) and len(v) == 3:
                        # 如果mask_values是RGB格式，但图像是灰度，跳过
                        continue
                    # 只映射非255的像素值（255保持为ignore_index）
                    if v != 255:
                        mask[img == v] = i
            elif img.ndim == 3:
                # RGB标签图像：使用颜色匹配
                if img.shape[2] == 3:  # RGB
                    for i, v in enumerate(mask_values):
                        if isinstance(v, list) and len(v) == 3:
                            # RGB颜色匹配，但跳过[255,255,255]
                            if v != [255, 255, 255]:
                                color_match = np.all(img == np.array(v), axis=2)
                                mask[color_match] = i
                        else:
                            # 如果mask_values不是颜色格式，尝试转换
                            continue
                else:
                    # 其他情况，取第一个通道
                    img_gray = img[:, :, 0]
                    for i, v in enumerate(mask_values):
                        if not isinstance(v, list) and v != 255:
                            mask[img_gray == v] = i
            
            # 特别处理黑色像素（通常是unlabelled）
            if img.ndim == 2:
                # 灰度图像中，0和255都视为unlabelled
                mask[img == 0] = 255
                # 255值保持不变（已经是ignore_index）
            elif img.ndim == 3:
                # RGB图像中，黑色像素视为unlabelled
                black_pixels = np.all(img == [0, 0, 0], axis=2)
                mask[black_pixels] = 255
                # 白色像素也视为unlabelled（可能来自填充）
                white_pixels = np.all(img == [255, 255, 255], axis=2)
                mask[white_pixels] = 255
            
            # 调试：检查mask的值范围
            unique_values = np.unique(mask)
            if len(unique_values) > 20 or (unique_values.max() > 255) or (unique_values.min() < 0):
                print(f"DEBUG: Unexpected mask values in preprocessing: {unique_values}")
                print(f"  Original image shape: {img.shape}, dtype: {img.dtype}")
                print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
                print(f"  mask_values length: {len(mask_values)}")
                print(f"  mask_values sample: {mask_values[:3] if len(mask_values) > 3 else mask_values}")
            
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0
            
            # 可选的均值归一化
            if normalize and mean is not None and std is not None:
                mean = np.array(mean).reshape(-1, 1, 1)
                std = np.array(std).reshape(-1, 1, 1)
                img = (img - mean) / std

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix +'_L' + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # 确保mask是正确的格式（单通道灰度图或RGB）
        # 如果mask是RGB格式，保持RGB；如果是灰度，保持灰度
        if mask.mode not in ['L', 'RGB']:
            mask = mask.convert('RGB')  # 转换为RGB以保持兼容性
        
        # 应用数据增强
        img, mask = self.apply_augmentation(img, mask)

        img_processed = self.preprocess(self.mask_values, img, self.scale, is_mask=False, 
                                       normalize=self.normalize, mean=self.mean, std=self.std)
        mask_processed = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        result = {
            'image': torch.as_tensor(img_processed.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_processed.copy()).long().contiguous()
        }
        
        # 可选：返回原始图像用于可视化（但会降低训练速度）
        if self.return_raw:
            # 转换为tensor以便DataLoader处理
            img_array = np.array(img)
            if img_array.ndim == 3:
                img_array = img_array.transpose((2, 0, 1))  # HWC -> CHW
            result['raw_image'] = torch.as_tensor(img_array.copy()).float()
            
        return result

