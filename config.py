from dataclasses import dataclass, field, asdict
from typing import Tuple
import json

@dataclass
class DiffusionConfig:
    num_timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02

@dataclass
class UNetConfig:
    in_channels = 3
    base_channels = 64
    channel_mults = (1,2,4)
    num_res_blocks = 2
    attn_resolutions = (16,)
    time_dim = 256
    dropout = 0.1
    num_heads = 4
    num_groups = 32
    image_size = 32

@dataclass
class DiTConfig:
    image_size = 32
    patch_size = 4
    in_channels = 3
    hidden_dim = 256
    depth = 8
    num_heads = 4
    mlp_ratio = 4.0
    time_dim = 256
    dropout = 0.0

@dataclass
class TrainConfig:
    batch_size = 128
    learning_rate = 1e-4
    weight_decay = 0.0
    num_epochs = 100
    grad_clip = 1.0
    ema_decay = 0.9999
    log_every = 100
    sample_every = 5000
    save_every = 10000
    num_sample_images = 64
    seed = 67