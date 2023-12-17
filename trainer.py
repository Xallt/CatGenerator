from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
from models.vae import VAE


class Trainer:
    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        lr: float,
        weight_decay: float,
        device: torch.device = torch.device("cpu"),
        num_epochs=100,
        log_every=100,
        save_every=1000,
        checkpoint_file="checkpoint.pt",
        num_samples=4,
    ):
        self.model = model

        self.train_dl = train_dl
        self.val_dl = val_dl
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.init_lr = lr
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
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
        kwargs = {}
        # Behavior specific to VAE training
        if isinstance(self.model, VAE):
            kwargs["kld_weight"] = 0.02
        loss_dict = self.model.compute_loss(x.to(self.device), **kwargs)
        loss = loss_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

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
            dl = self.train_dl
            if progress_bar:
                dl = tqdm(dl)
            loss_dict_sum = defaultdict(float)
            for x in dl:
                loss_dict = self._run_step(x)

                curr_count += len(x)
                for k, v in loss_dict.items():
                    loss_dict_sum[k] += v * len(x)

                if (step + 1) % self.log_every == 0:
                    for k, v in loss_dict_sum.items():
                        writer.add_scalar(f"train/{k}", v / curr_count, step)
                        loss_dict_sum[k] = 0
                    curr_count = 0
                if (step + 1) % self.save_every == 0:
                    torch.save(self.model.state_dict(), self.checkpoint_file)
                    self.model.eval()
                    with torch.no_grad():
                        samples = self.model.sample(self.num_samples)
                    writer.add_images("sample", samples.cpu() * 0.5 + 0.5, step)
                    self.model.train()
                step += 1
