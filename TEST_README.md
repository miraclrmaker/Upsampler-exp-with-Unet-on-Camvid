# CamVid UNet 测试脚本使用指南

##  功能介绍

这个测试脚本 `test.py` 提供了完整的模型评估功能：

1. **智能模型选择**：从最后N个checkpoint中自动选择验证集表现最佳的模型
2. **全面指标计算**：计算像素准确度、mIoU、每类像素准确度等详细指标
3. **结果保存**：自动保存详细的评估结果到CSV文件
4. **可视化报告**：生成易读的评估报告

## 使用

```bash
python test.py \
    --batch-size 4 \
    --n-models 10 \
    --scale 1.0 \
    --amp \
    --normalize \
    --save-results ./test_results/ \
    --checkpoint-dir ./checkpoints/
```

## 输出结果

脚本会生成以下文件：

```
test_results/
├── overall_metrics.csv        # 总体指标
├── per_class_metrics.csv      # 每类别指标
├── confusion_matrix.csv       # 混淆矩阵
└── model_info.csv            # 最佳模型信息
```

## 评估指标

### 总体指标
- **Mean Dice Score**: Dice系数均值
- **Overall Pixel Accuracy**: 总体像素准确度
- **Mean IoU (mIoU)**: 平均交并比
- **Frequency Weighted IoU**: 频率加权IoU

### 每类别指标
- **Per-class Pixel Accuracy**: 每类别像素准确度
- **Per-class IoU**: 每类别交并比

### CamVid类别
1. Sky (天空)
2. Building (建筑)
3. Pole (杆子)
4. Road (道路)
5. Sidewalk (人行道)
6. Tree (树木)
7. SignSymbol (标志符号)
8. Fence (围栏)
9. Car (汽车)
10. Pedestrian (行人)
11. Bicyclist (骑行者)

## 参数说明

- `--batch-size`: 测试批次大小（默认：4）
- `--n-models`: 考虑的最后N个模型数量（默认：10）
- `--scale`: 图像缩放因子（默认：1.0）
- `--amp`: 启用混合精度
- `--normalize`: 启用ImageNet归一化
- `--save-results`: 结果保存目录
- `--checkpoint-dir`: checkpoint目录路径
