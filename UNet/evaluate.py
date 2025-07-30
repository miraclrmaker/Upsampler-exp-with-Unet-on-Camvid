import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_inter = None  # 用于mIoU
    total_union = None
    total_correct = None  # 用于每类像素准确度
    total_label = None

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred_bin = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred_bin, mask_true, reduce_batch_first=False)
                # mIoU & per-class acc for binary
                pred = mask_pred_bin.long().squeeze(1)
                label = mask_true
                n_class = 2
            else:
                # 检查mask值的有效性：应该在[0, n_classes)范围内，或者等于255(ignore_index)
                valid_mask_values = (mask_true >= 0) & (mask_true < net.n_classes) | (mask_true == 255)
                assert valid_mask_values.all(), f'True mask indices should be in [0, {net.n_classes}[ or 255 for ignore, but found min={mask_true.min()}, max={mask_true.max()}'
                pred = mask_pred.argmax(dim=1)
                label = mask_true
                n_class = net.n_classes
                
                # 现在很简单：只有11个有效类别，unlabelled像素值为255
                # 创建有效像素的mask（排除ignore_index=255）
                valid_mask = (mask_true != 255)
                if valid_mask.sum() > 0:
                    # 对于dice计算，我们需要在原始空间形状上操作
                    # 创建一个临时mask，将ignore像素设为0类别用于one-hot编码
                    temp_true = mask_true.clone()
                    temp_pred = pred.clone()
                    temp_true[mask_true == 255] = 0  # 临时将ignore像素设为类别0
                    temp_pred[mask_true == 255] = 0  # 预测也对应设为类别0
                    
                    # 创建one-hot编码（保持原始空间维度）
                    mask_true_oh = F.one_hot(temp_true, net.n_classes).float()  # [B, H, W, C]
                    mask_pred_oh = F.one_hot(temp_pred, net.n_classes).float()  # [B, H, W, C]
                    
                    # 转换为 [B, C, H, W] 格式
                    mask_true_oh = mask_true_oh.permute(0, 3, 1, 2)
                    mask_pred_oh = mask_pred_oh.permute(0, 3, 1, 2)
                    
                    # 应用valid_mask到每个类别通道
                    valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(mask_true_oh)
                    mask_true_oh = mask_true_oh * valid_mask_expanded.float()
                    mask_pred_oh = mask_pred_oh * valid_mask_expanded.float()
                    
                    # 计算所有11个类别的dice
                    dice_score += multiclass_dice_coeff(
                        mask_pred_oh, 
                        mask_true_oh, 
                        reduce_batch_first=False
                    )

            # mIoU & per-class acc - 只计算有效像素（排除255）
            pred_flat = pred.view(-1)
            label_flat = label.view(-1)
            inter = torch.zeros(n_class, device=device)
            union = torch.zeros(n_class, device=device)
            correct = torch.zeros(n_class, device=device)
            total = torch.zeros(n_class, device=device)
            
            # 只对非忽略像素计算统计
            valid_pixels = (label_flat != 255)  # 排除ignore_index像素
            if valid_pixels.sum() > 0:
                valid_pred = pred_flat[valid_pixels]
                valid_label = label_flat[valid_pixels]
                
                for cls in range(n_class):
                    pred_i = (valid_pred == cls)
                    label_i = (valid_label == cls)
                    inter[cls] = (pred_i & label_i).sum()
                    union[cls] = (pred_i | label_i).sum()
                    correct[cls] = (pred_i & label_i).sum()
                    total[cls] = label_i.sum()
            if total_inter is None:
                total_inter = inter
                total_union = union
                total_correct = correct
                total_label = total
            else:
                total_inter += inter
                total_union += union
                total_correct += correct
                total_label += total

    net.train()
    mean_dice = dice_score / max(num_val_batches, 1)
    # mIoU - 现在直接计算所有11个类别
    iou = total_inter / (total_union + 1e-6)
    miou = iou.mean().item()
    # per-class pixel accuracy - 所有11个类别
    per_class_acc_all = (total_correct / (total_label + 1e-6))
    per_class_acc = per_class_acc_all.tolist()
    # 也返回每类别的IoU用于详细分析
    per_class_iou = iou.tolist()
    return mean_dice, miou, per_class_acc, per_class_iou
