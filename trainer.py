from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from diffusion import Diffusion


class Trainer:
    def __init__(
        self,
        diffusion: Diffusion,
        model: nn.Module,
        train_iter, # iterable that yields (x, y)
        lr: float,
        weight_decay: float,
        steps: int,
        device: torch.device = torch.device('cpu'),
        log_every=100,
        checkpoint_file='checkpoint.pt'
    ):
        self.diffusion = diffusion

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.log_every = log_every
        self.save_every = 1000
        self.checkpoint_file = checkpoint_file

    def _anneal_lr(self, step: int):
        """
        Performs annealing of lr.
        """

        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x: torch.FloatTensor, y):
        """
        A single training step.
        Calculates loss for a single batch. 
        Then performs a single optimizer step and returns loss.
        """
        loss = self.diffusion.train_loss(self.model, x.to(self.device), y.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run_loop(self):
        """
        Training loop.
        """
        step = 0
        curr_loss_gauss = 0.0

        writer = SummaryWriter()
        curr_count = 0
        loss_agg = 0
        for step in tqdm(range(self.steps)):
            x, y = next(self.train_iter)
            batch_loss = self._run_step(x, y)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_gauss += batch_loss * len(x)

            if (step + 1) % self.log_every == 0:
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                writer.add_scalar('train/loss', gloss, step)
                curr_count = 0
                curr_loss_gauss = 0.0
            if (step + 1) % self.save_every == 0:
                torch.save(self.model.state_dict(), self.checkpoint_file)