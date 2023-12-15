import torch
import torch.nn as nn
import math
from .blocks import SinusoidalPosEmb, ResnetBlock

# Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, levels, initial_features=64, timesteps=1000, time_emb_dim=128):
        super().__init__()
        self.levels = levels
        self.timesteps = timesteps

        # Positional embedding for time
        self.time_emb = SinusoidalPosEmb(dim=time_emb_dim, theta=10000)

        # Initial convolution layer
        self.init_conv = nn.Conv2d(in_channels, initial_features, kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in range(levels):
            in_dim = initial_features * (2 ** level)
            out_dim = initial_features * (2 ** (level + 1))

            self.down_blocks.append(
                ResnetBlock(dim=in_dim, dim_out=out_dim, time_emb_dim=time_emb_dim)
            )

            # Upsampling blocks should have reverse order and dimensions
            self.up_blocks.insert(
                0, ResnetBlock(dim=out_dim * 2, dim_out=in_dim, time_emb_dim=time_emb_dim)
            )

        # Final convolution to get to the output channels
        self.final_conv = nn.Conv2d(initial_features, out_channels, kernel_size=1)

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_emb(t)

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling
        intermediates = []
        for down in self.down_blocks:
            x = down(x, t_emb)
            intermediates.append(x)
            x = nn.functional.avg_pool2d(x, kernel_size=2)

        # Upsampling
        for up, intermediate in zip(self.up_blocks, reversed(intermediates)):
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, intermediate], dim=1)
            x = up(x, t_emb)

        # Final convolution
        return self.final_conv(x)
