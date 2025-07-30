# U-Net 语义分割项目

## 文件说明

​	训练模型在文件夹下 train.py 文件中，需要修改一下数据集的路径，我已经写好了相对路径，只需要把第25行代码取消注释，并把26行代码注释掉就可以了。然后就可以直接训练

​	训练过程中会自动进行 evaluate，所以 evaluate 的过程可以直接看终端产生的记录，也可以进一步使用 **TensorBoard** 查看模型在训练过程中的详细数据

## 日志系统（TensorBoard）

本项目使用TensorBoard来记录和可视化训练过程。

### 查看训练日志
```bash
tensorboard --logdir=./logs/tensorboard --port=6006
```
然后在浏览器中打开 `http://localhost:6006` 查看训练日志。

### TensorBoard功能

TensorBoard会记录以下信息：

1. **训练指标**：
   - 训练损失 (Train/Loss)
   - 学习率 (Learning_Rate)

2. **验证指标**：
   - Dice系数 (Validation/Dice)
   - 平均IoU (Validation/mIoU)
   - 平均每类像素精度 (Validation/mean_per_class_acc)
   - 每个类别的精度 (Validation/per_class_acc_*)
   - 每个类别的IoU (Validation/per_class_iou_*)

3. **图像可视化**：
   - 输入图像 (Images/Input)
   - 真实标签 (Images/True_Mask)
   - 预测结果 (Images/Pred_Mask)

4. **模型参数**：
   - 权重分布直方图 (Weights/*)

5. **最终结果**：
   - 最佳分数 (Final/Best_Score)

### CamVid数据集信息

CamVid数据集包含11个语义类别：
- Sky (天空)
- Building (建筑)
- Pole (杆子)
- Road (道路)
- Sidewalk (人行道)
- Tree (树木)
- SignSymbol (标志符号)
- Fence (围栏)
- Car (汽车)
- Pedestrian (行人)
- Bicyclist (骑自行车的人)

## 使用方法

1. 确保数据集放在正确的目录：
   - 训练图像：`CamVid/train/`
   - 训练标签：`CamVid/train_labels/`
   - 验证图像：`CamVid/val/`
   - 验证标签：`CamVid/val_labels/`

2. 运行训练：
   ```bash
   python UNet/train.py --epochs 100 --batch-size 8 --learning-rate 3e-5
   ```
