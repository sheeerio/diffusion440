import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Optional
import math

from utils.blocks import SinPositionEmbedding

class PatchEmbed(nn.Module):
    """
    split image into non-overlapping patches and linearly embed them
    e.g. 32x32 image with patch_size=4: num_patches = (32/4)^2 = 64 patches; 
                                each patch is 4x4x3, projected to hidden_dim
    """
    def __init__(self, image_size=32, patch_size=4, in_channels=3, hidden_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2) # B,num_patches,hidden_dim

class UnpatchEmbed(nn.Module):
    def __init__(self, image_size=32, patch_size=4, out_channels=3, hidden_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.proj = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        self.out_channels = out_channels
    
    def forward(self, x):
        B, N, D = x.shape
        x = self.proj(x)
        p, c, g = self.patch_size, self.out_channels, self.grid_size
        
        x = x.reshape(B, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4) # B,C,g,p,g,p
        x = x.reshape(B, c, g * p, g * p) # B,C,H,W
        return x

class AdaLNZero(nn.Module):
    """
    ådaptive layernorm with zero initialization (from DiT paper)
    predicts gamma1, gamma2, beta1, beta2, alpha1, alpha2 from conditioning vector c
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim)
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)
    
    def forward(self, c):
        params = self.proj(c)
        return params.chunk(6, dim=-1)