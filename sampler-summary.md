# CamVid-UNet实验 - 不同算子性能信息汇总


| 算子组合 | 下采样方式 | 上采样方式 | 最佳轮次 | 最终测试mIoU | 测试Dice | 测试像素准确率 |
|---------|-----------|-----------|----------|-------------|----------|-------------|
| max-pooling_bilinear | max-pooling | bilinear | 115 | 0.6403 | 0.6516 | 0.9036 |
| max-pooling_carafe | max-pooling | carafe | 11 | 0.6211 | 0.6801 | 0.9036 |
| max-pooling_deconvolution | max-pooling | deconvolution | 118 | 0.6492 | 0.6555 | 0.9056 |
| max-pooling_nearest | max-pooling | nearest | 115 | 0.6491 | 0.6608 | 0.9063 |
| max-pooling-indices_maxunpooling | max-pooling-indices | maxunpooling | 119 | 0.6375 | 0.6385 | 0.9033 |
| max-pooling_dysample_lp | max-pooling | dysample_lp | 115 | 0.6574 | 0.6638 | 0.9088 |
| max-pooling_dysample_lp-dynamic | max-pooling | dysample_lp-dynamic | 113 | 0.6492 | 0.6623 | 0.9043 |
| max-pooling_dysample_pl | max-pooling | dysample_pl | 118 | 0.6481 | 0.6586 | 0.9054 |
| max-pooling_dysample_pl-dynamic | max-pooling | dysample_pl-dynamic | 118 | 0.6452 | 0.6486 | 0.9040 |


### mIoU性能排名（由高到低）
1. **max-pooling_dysample_lp**: 0.6574
2. **max-pooling_deconvolution**: 0.6492
3. **max-pooling_dysample_lp-dynamic**: 0.6492
4. **max-pooling_nearest**: 0.6491
5. **max-pooling_dysample_pl**: 0.6481
6. **max-pooling_dysample_pl-dynamic**: 0.6452
7. **max-pooling_bilinear**: 0.6403
8. **max-pooling-indices_maxunpooling**: 0.6375
9. **max-pooling_carafe**: 0.6211

### Dice系数排名（由高到低）
1. **max-pooling_carafe**: 0.6801
2. **max-pooling_dysample_lp**: 0.6638
3. **max-pooling_dysample_lp-dynamic**: 0.6623
4. **max-pooling_nearest**: 0.6608
5. **max-pooling_dysample_pl**: 0.6586
6. **max-pooling_deconvolution**: 0.6555
7. **max-pooling_bilinear**: 0.6516
8. **max-pooling_dysample_pl-dynamic**: 0.6486
9. **max-pooling-indices_maxunpooling**: 0.6385

### 像素准确率排名（由高到低）
1. **max-pooling_dysample_lp**: 0.9088
2. **max-pooling_nearest**: 0.9063
3. **max-pooling_deconvolution**: 0.9056
4. **max-pooling_dysample_pl**: 0.9054
5. **max-pooling_dysample_lp-dynamic**: 0.9043
6. **max-pooling_dysample_pl-dynamic**: 0.9040
7. **max-pooling_bilinear**: 0.9036
8. **max-pooling_carafe**: 0.9036
9. **max-pooling-indices_maxunpooling**: 0.9033