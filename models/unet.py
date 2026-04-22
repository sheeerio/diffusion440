import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import List, Optional

from utils.blocks import TimeMLP

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, time_dim=256):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, num_channels), num_channels)
        self.proj = nn.Linear(time_dim, num_channels * 2)
    
    def forward(self, x, time_emb):
        x = self.norm(x)
        scale_shift = self.proj(time_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (1 + scale) + shift
    
class ResBlock(nn.Module):
    """
    x -> GN -> SiLU -> Conv -> [+ time] -> GN -> SiLU -> Dropout -> Conv -> [+ res]
    """
    def __init__(self, in_channels, out_channels, time_dim=256, dropout=0.1, num_groups=32):
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_channels, num_groups, time_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(out_channels, num_groups, time_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=3)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        if in_channels != out_channels: self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        else: self.residual_proj = nn.Identity()
    
    def forward(self, x, time_emb):
        residual = self.residual_proj(x)
        h = self.act(self.norm1(x, time_emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h ,time_em))
        h = self.conv2(self.dropout(h))
        return h + residual