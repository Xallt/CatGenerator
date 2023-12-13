import torch
import os
from torchvision.io import read_image
import numpy as np
from glob import glob

class CatDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, seed=42):
        self.path = path
        self.files = list(glob(os.path.join(path, '*.jpg')))
        self.transform = transform

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.train_size = int(0.8 * len(self.files))
        indices = np.arange(len(self.files))
        self.rng.shuffle(indices)
        self.train_indices = indices[:self.train_size]
        self.test_indices = indices[self.train_size:]

    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = read_image(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def train_loader(self, batch_size=1):
        return torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self.train_indices)
        )

    def test_loader(self, batch_size=1):
        return torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size,
            sampler=self.test_indices
        )