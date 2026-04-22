import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinPositionEmbedding(nn.Module):
    """
    to map timesteps to vectors; 
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(1000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1: emb = F.pad(emb, (0, 1))
        return emb

class TimeMLP(nn.Module):
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.sin = SinPositionEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, t):
        return self.mlp(self.sin(t))