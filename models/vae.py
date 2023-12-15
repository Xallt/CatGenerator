import torch
import torch.nn as nn
from .blocks import Block, ResnetBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_levels, num_blocks_per_level=2, latent_dim=256):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        layers = []
        channels = base_channels
        for level in range(num_levels):
            channels_next = base_channels * (2 ** level)
            for _ in range(num_blocks_per_level):
                layers.append(ResnetBlock(channels, channels_next))
                channels = channels_next
            if level < num_levels - 1:
                layers.append(nn.AvgPool2d(kernel_size=2))

        self.features = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(channels, latent_dim, kernel_size=1)
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.fc_mu = nn.Linear(latent_dim * (2 ** (num_levels - 1)) ** 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * (2 ** (num_levels - 1)) ** 2, latent_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.final_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def sample_from(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    def sample(self, x):
        mu, logvar = self(x)
        return self.sample_from(mu, logvar)
       

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, base_channels, num_levels, num_blocks_per_level=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_channels * (2 ** (num_levels - 1)) ** 2)
        self.reshape = Rearrange('b (c h w) -> b c h w', c=base_channels * (2 ** (num_levels - 1)), h=2 ** (num_levels - 1), w=2 ** (num_levels - 1))

        layers = []
        for level in reversed(range(num_levels)):
            channels = base_channels * (2 ** level)
            for _ in range(num_blocks_per_level):
                layers.append(ResnetBlock(channels, channels))
            if level > 0:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        self.features = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.features(x)
        x = self.final_conv(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels, in_resolution, num_levels, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.in_resolution = in_resolution
        self.num_levels = num_levels

        self.encoder = Encoder(in_channels, 32, num_levels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim, in_channels, 32, num_levels)
    
    def sample(self, num_samples=1, latent=None):
        if latent is None:
            latent = torch.randn(num_samples, self.latent_dim, device=self.device)

        return self.decoder(latent)