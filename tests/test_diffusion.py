import unittest
from models.unet import UNet
from diffusion import Diffusion
from trainer import Trainer
from diffusion import get_named_beta_schedule
import torch

class TestDiffusion(unittest.TestCase):

    def test_simple_diffusion_training(self):
        model = UNet(
            in_channels=3,
            out_channels=3,
            levels=0,
            time_emb_dim=16,
            initial_features=16,
        ).cuda()

        diffusion = Diffusion(
            model,
            betas=get_named_beta_schedule('sigmoid', 1000),
            mode="pred_start",
        )

        trainer = Trainer(
            diffusion,
            [torch.randn(1, 3, 32, 32)],
            lr=1e-3,
            weight_decay=1e-4,
            num_epochs=1,
            device='cuda',
            checkpoint_file='checkpoints/test.pt'
        )

        trainer.run_loop(progress_bar=False)