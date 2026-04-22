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

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Moduel):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x):
        B,C,H,W = x.shape
        residual = x 
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2] # B,heads,C//heads,HW
        q = q.permute(0, 1, 3, 2) # B,heads,HW,C//heads
        k = k # B,heads,C//heads,HW
        v = v.permute(0, 1, 3, 2) # B,heads,HW,C//heads

        attn = torch.matmul(q, k) * self.scale # B,heads,HW,HW
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v) # B,heads,HW,C//heads
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj_out(out)
        
        return out + residual
