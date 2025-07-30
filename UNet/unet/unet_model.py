""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, style1='max-pooling', style2='bilinear'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.style1 = style1
        self.style2 = style2

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, style1))
        self.down2 = (Down(128, 256, style1))
        self.down3 = (Down(256, 512, style1))
        factor = 1 if self.style2=='deconvolution' else 2
        self.down4 = (Down(512, 1024 // factor, style1))
        self.up1 = (Up(1024, 512 // factor, style2))
        self.up2 = (Up(512, 256 // factor, style2))
        self.up3 = (Up(256, 128 // factor, style2))
        self.up4 = (Up(128, 64, style2))
        self.outc = (OutConv(64, n_classes))
        
        # 用于特征可视化的标志
        self.feature_extraction_mode = False
        self.extracted_features = {}

    def forward(self, x):
        x1 = self.inc(x)
        
        # Handle different down-sampling styles
        if self.style1 == 'max-pooling-indices' and self.style2 == 'maxunpooling':
            # For MaxUnpooling, we need to store indices
            x2, indices1 = self.down1(x1)
            x3, indices2 = self.down2(x2)
            x4, indices3 = self.down3(x3)
            x5, indices4 = self.down4(x4)
            
            # Use indices for MaxUnpooling
            x = self.up1(x5, x4, indices4)
            x = self.up2(x, x3, indices3)
            x = self.up3(x, x2, indices2)
            x = self.up4(x, x1, indices1)
        else:
            # Standard downsampling without indices
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Standard upsampling
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
    
    def forward_with_features(self, x):
        """前向传播并返回中间特征图"""
        features = {}
        
        x1 = self.inc(x)
        features['inc'] = x1
        
        # Handle different down-sampling styles for feature extraction
        if self.style1 == 'max-pooling-indices' and self.style2 == 'maxunpooling':
            # For MaxUnpooling, we need to store indices
            x2, indices1 = self.down1(x1)
            features['down1'] = x2
            
            x3, indices2 = self.down2(x2)
            features['down2'] = x3
            
            x4, indices3 = self.down3(x3)
            features['down3'] = x4
            
            x5, indices4 = self.down4(x4)
            features['down4'] = x5
            
            # Use indices for MaxUnpooling
            x = self.up1(x5, x4, indices4)
            features['up1'] = x
            
            x = self.up2(x, x3, indices3)
            features['up2'] = x
            
            x = self.up3(x, x2, indices2)
            features['up3'] = x
            
            x = self.up4(x, x1, indices1)
            features['up4'] = x
        else:
            # Standard downsampling without indices
            x2 = self.down1(x1)
            features['down1'] = x2
            
            x3 = self.down2(x2)
            features['down2'] = x3
            
            x4 = self.down3(x3)
            features['down3'] = x4
            
            x5 = self.down4(x4)
            features['down4'] = x5
            
            # Standard upsampling
            x = self.up1(x5, x4)
            features['up1'] = x
            
            x = self.up2(x, x3)
            features['up2'] = x
            
            x = self.up3(x, x2)
            features['up3'] = x
            
            x = self.up4(x, x1)
            features['up4'] = x
        
        logits = self.outc(x)
        features['outc'] = logits
        
        return logits, features
    
    def get_feature_shapes(self, input_shape):
        """获取各层特征图的形状信息（用于调试）"""
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)
            _, features = self.forward_with_features(dummy_input)
            
            shapes = {}
            for name, feature in features.items():
                shapes[name] = feature.shape
            
            return shapes

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)