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

class DiTBlock(nn.Module):
    """
    x -> ada layernorm -> multihead selfattn -> gate -> residual
    x -> ada layernorm -> mlp -> gate -> residual
    """
    def __init__(
        self,
        hidden_dim,
        num_heads = 4,
        mlp_ratio = 4.0,
        cond_dim = 256,
        dropout = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout))
        )
        self.adaln = AdaLNZero(hidden_dim, cond_dim)

    def forward(self, x, c):
        """
        x: (B, N, D) - token sequence
        c: (B, cond_dim) - conditioning (timestep embedding)
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(c)
        # attention
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + alpha1.unsqueeze(1) * h 
        # mlp
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.mlp(h)
        x = x + alpha2.unsqueeze(1) * h 
        return x

    class DiT(nn.Module):
        def __init__(
            self,
            image_size=32,
            patch_size=4,
            in_channels=3,
            hidden_dim=256,
            depth=8,
            num_heads=4,
            mlp_ratio=4.0,
            time_dim=256,
            dropout=0.0
        ):
            super().__init__()
            self.image_size = image_size
            self.patch_size = patch_size
            self.in_channels = in_channels
            self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_dim)
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.time_embed = nn.Sequential(
                SinPositionEmbedding(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )

            self.blocks = nn.ModuleList([
                DiTBlock(hidden_dim, num_heads, mlp_ratio, time_dim, dropout) for _ in range(depth)
            ])
            self.final_form = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.final_adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, 2 * hidden_dim)
            )
            nn.init.zeros_(self.final_adaln[1].weight)
            nn.init.zeros_(self.final_adaln[1].bias)
            self.unpatch = UnpatchEmbed(image_size, patch_size, in_channels, hidden_dim)
        
        def forward(self, x, t):
            c = self.time_embed(t)
            tokens = self.patch_embed(x) + self.pos_embed # B.N.D
            for block in self.blocks:
                tokens = block(tokens, c)
            shift_scale = self.final_adaln(c)
            shift, scale = shift_scale.chunk(2, dim=-1)
            tokens = self.final_norm(tokens)
            tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            return self.unpatch(tokens)

        def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)