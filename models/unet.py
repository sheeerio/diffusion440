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


class Unet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        channel_mults=(1,2,4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        time_dim=256,
        dropout=0.1,
        num_heads=4,
        num_groups=32,
        image_size=32
    ):
        super().__init__()
        self.image_size = image_size
        self.time_embed = TimeMLP(time_dim, time_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch = base_channels
        self.skip_channels = [ch]
        current_res = image_size

        for level_idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout, num_groups)])
                if current_res in attn_resolutions: block.append(SelfAttention2d(out_ch, num_heads, num_groups))
                level_blocks.append(block)
                ch = out_ch
                self.skip_channels.append(ch)
            self.encoder.append(level_blocks)

            if level_idx < len(channel_mults) - 1:
                self.pool.append(Downsample(ch))
                self.skip_channels.append(ch)
                current_res //= 2
        
        self.mid1 = ResBlock(ch, ch, time_dim, dropout, num_groups)
        self.mid_attn = SelfAttention2d(ch, num_heads, num_groups)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout, num_groups)
        
        # decoder bits now (we be up now)
        self.decoder = nn.ModuleList()
        self.up = nn.ModuleList()

        for level_idx in reversed(range(len(channel_mults))):
            mult = channel_mults[level_idx]
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()

            for i in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                block = nn.ModuleList([ResBlock(ch + skip_ch, out_ch, time_dim, dropout, num_groups)])
                enc_res = imagee_size // (2 ** level_idx) # more like unc_res amirite ahaahahhaahahahah; checks if encoder had attn here
                if enc_res in attn_resolutions: block.append(SelfAttention2d(out_ch, num_heads, num_groups))
                level_blocks.append(block)
                ch = out_ch
            self.decoder.append(level_blocks)
            if level_idx > 0: self.up.append(Upsample(ch))
        
        self.out_norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init_zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t):
        time_emb = self.time_embed(t)
        h = self.input_conv(x)
        skips = [h]

        for level_idx, level_blocks in enumerate(self.encoder):
            for block in level_blocks:
                for layer in block:
                    h = layer(h, t_emb) if if isinstance(layer, ResBlock) else layer(h)
                skips.append(h)
            if level_idx < len(self.pool):
                h = self.pool[level_idx](h)
                skips.append(h)
        
        h = self.mid1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, time_emb)

        for level_idx, level_blocks in enumerate(self.decoder):
            for block in level_blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                for layer in block:
                    h = layer(h, time_emb) if isinstance(layer, ResBlock) else layer(h)
            if level_idx < len(self.up): h = self.up[level_idx](h)
        
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)