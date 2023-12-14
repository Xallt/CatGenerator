from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from diffusion import Diffusion
from tqdm.auto import tqdm
import numpy as np


class Trainer:
    def __init__(
        self,
        diffusion: Diffusion,
        model: nn.Module,
        dl,
        lr: float,
        weight_decay: float,
        device: torch.device = torch.device('cpu'),
        num_epochs=100,
        log_every=100,
        save_every=1000,
        checkpoint_file='checkpoint.pt'
    ):
        self.diffusion = diffusion

        self.dl = dl
        self.num_epochs = num_epochs
        self.init_lr = lr
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_file = checkpoint_file

    def _run_step(self, x):
        """
        A single training step.
        Calculates loss for a single batch. 
        Then performs a single optimizer step and returns loss.
        """
        loss = self.diffusion.train_loss(self.model, x.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run_loop(self, progress_bar=True):
        """
        Training loop.
        """
        step = 0
        curr_loss_gauss = 0.0

        writer = SummaryWriter()
        curr_count = 0
        loss_agg = 0
        self.model.train()
        for epoch_num in range(self.num_epochs):
            dl = self.dl
            if progress_bar:
                dl = tqdm(dl)
            for x in dl:
                batch_loss = self._run_step(x)

                curr_count += len(x)
                curr_loss_gauss += batch_loss * len(x)

                if (step + 1) % self.log_every == 0:
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    writer.add_scalar('train/loss', gloss, step)
                    curr_count = 0
                    curr_loss_gauss = 0.0
                if (step + 1) % self.save_every == 0:
                    torch.save(self.model.state_dict(), self.checkpoint_file)
                step += 1