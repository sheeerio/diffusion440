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

@dataclass
class EvalConfig:
    eval_timesteps = (100,  250, 500, 750)
    patch_sizes = (4, 8)
    mask_ratios = (0.25, 0.5, 0.75)
    block_fractions = (0.25, 0.5)
    num_eval_batches = 10
    eval_batch_size = 64

@dataclass
class ExperimentConfig:
    name = "dit_vs_unet"
    model_type = "unet"
    diffusion = field(default_factory=DiffusionConfig)
    unet = field(default_factory=UNetConfig)
    dit = field(default_factory=DiTConfig)
    train = field(default_factory=TrainConfig)
    eval = field(default_factory=EvalConfig)
    output_dir = "./outputs"
    device = "cuda"

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(
            name=d['name'],
            model_type=d['model_type'],
            diffusion=DiffusionConfig(**d['diffusion']),
            unet=UNetConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d['unet'].items()}),
            dit=DiTConfig(**d['dit']),
            train=TrainConfig(**d['train']),
            eval=EvalConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d['eval'].items()}),
            output_dir=d['output_dir'],
            device=d['device'],
        )