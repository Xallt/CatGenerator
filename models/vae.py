import torch
import torch.nn as nn
from .blocks import Block, ResnetBlock
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_resolution,
        base_channels,
        num_levels,
        num_blocks_per_level=2,
        latent_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        assert (
            type(in_resolution) == int
        ), f"Input resolution must be an integer, got {in_resolution}"
        assert (
            in_resolution % (2**num_levels) == 0
        ), f"Resolution {in_resolution} is not divisible by 2^{num_levels}"
        assert (
            latent_dim % (in_resolution // (2**num_levels)) == 0
        ), f"Latent dim {latent_dim} is not divisible by {in_resolution // (2 ** num_levels)}"

        self.in_resolution = in_resolution
        self.num_levels = num_levels
        self.latent_dim = latent_dim
        self.final_resolution = in_resolution // (2**num_levels)

        self.initial_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        layers = []
        channels = base_channels
        for level in range(num_levels + 1):
            channels_next = channels * 2 if level != num_levels else channels
            for _ in range(num_blocks_per_level):
                layers.append(ResnetBlock(channels, channels_next, dropout=dropout))
                channels = channels_next
            if level != num_levels:
                layers.append(nn.AvgPool2d(kernel_size=2))

        self.features = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(channels, latent_dim, kernel_size=1)
        self.fc_mu = nn.Linear(latent_dim * self.final_resolution**2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * self.final_resolution**2, latent_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
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
    def __init__(
        self,
        latent_dim,
        starting_resolution,
        base_channels,
        out_channels,
        num_levels,
        num_blocks_per_level=2,
        dropout=0.0,
    ):
        super().__init__()
        self.starting_resolution = starting_resolution
        self.fc = nn.Linear(latent_dim, latent_dim * (starting_resolution**2))
        self.start_conv = nn.Conv2d(latent_dim, base_channels, kernel_size=1)
        self.latent_dim = latent_dim

        layers = []
        channels = base_channels
        for level in range(num_levels + 1):
            channels_next = channels // 2 if level != 0 else channels
            for _ in range(num_blocks_per_level):
                layers.append(ResnetBlock(channels, channels_next, dropout=dropout))
                channels = channels_next
            if level != num_levels:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.features = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(
            x.shape[0],
            self.latent_dim,
            self.starting_resolution,
            self.starting_resolution,
        )
        x = self.start_conv(x)
        x = self.features(x)
        x = self.final_conv(x)
        return x


class VAE(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels,
        in_resolution,
        num_levels,
        latent_dim,
        num_blocks_per_level=2,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_resolution = in_resolution
        self.final_resolution = in_resolution // (2**num_levels)
        self.num_levels = num_levels

        self.encoder = Encoder(
            in_channels,
            in_resolution,
            base_channels,
            num_levels,
            latent_dim=latent_dim,
            num_blocks_per_level=num_blocks_per_level,
            dropout=dropout,
        )
        self.decoder = Decoder(
            latent_dim,
            self.final_resolution,
            base_channels * (2**num_levels),
            in_channels,
            num_levels,
            num_blocks_per_level=num_blocks_per_level,
            dropout=dropout,
        )

    def sample(self, num_samples=1, latent=None, device="cuda"):
        if latent is None:
            latent = torch.randn(num_samples, self.encoder.latent_dim, device=device)

        return self.decoder(latent)

    def compute_loss(self, x, kld_weight=1.0):
        mu, logvar = self.encoder(x)
        encoder_latent_sample = self.encoder.sample_from(mu, logvar)
        decoded_x = self.decoder(encoder_latent_sample)

        reconstruction_loss = F.mse_loss(x, decoded_x)
        KLd_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = reconstruction_loss + KLd_loss * kld_weight

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "KLd_loss": KLd_loss,
            "x": x,
            "x_encoded": encoder_latent_sample,
            "x_decoded": decoded_x,
        }
