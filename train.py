import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from config import DiffusionConfig, UNetConfig, DiTConfig, TrainConfig
from models.unet import Unet
from models.dit import DiT
from utils.diffusion import DiffusionSchedule, compute_loss, ddim_sample

def load_cifar10(batch_size, train=True):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        Transforms.ToTensor(),
        Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = datasets.CIFAR10(root="../../data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, drop_last=train)

