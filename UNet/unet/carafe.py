import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class CARAFE_upsample(nn.Module):
    def __init__(self, c_in, c_mid, encoder_size=5, ratio=2, k_up=5):
        #默认编码器kernel大小为5
        super().__init__()
        padding = (encoder_size - 1) // 2 if encoder_size % 2 == 0 else encoder_size // 2
        self.ratio = ratio
        self.k_up = k_up
        self.unfold_padding = (k_up - 1) // 2 if k_up % 2 == 0 else k_up // 2
        self.comp = nn.Conv2d(c_in, c_mid, kernel_size=1, padding=0)
        self.content_encoder = nn.Conv2d(c_mid, (ratio * k_up) ** 2,
                                         kernel_size=encoder_size,
                                         padding=padding)

    def forward(self, x):
        _, _, _, W = x.size()
        y = self.comp(x)
        y = self.content_encoder(y)
        y = einops.rearrange(y, 'b (r1 r2 k_up2) h w -> b k_up2 (h r1) (w r2)',
                             r1=self.ratio, r2=self.ratio)

        z = F.unfold(x, kernel_size=self.k_up, padding=self.unfold_padding)
        z = einops.rearrange(z, 'b (c k_up2) (h w) -> b k_up2 c h w',
                             k_up2=self.k_up ** 2, w=W)
        z = einops.repeat(z, 'b k c h w -> ratio_2 b k c h w', ratio_2=self.ratio ** 2)
        z = einops.rearrange(z, '(r1 r2) b k_up2 c h w -> b k_up2 c (h r1) (w r2)',
                             r1=self.ratio)
        x = torch.einsum('bkchw,bkhw->bchw', [z, y])
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.Tensor(3, 16, 24, 24)
    carafe = CARAFE_upsample(16, 64, 5, ratio=3)
    oup = carafe(x)
    print(oup.size())