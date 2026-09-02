import math,torch
import torch.nn as nn
import torch.nn
from dataclasses import dataclass

from mamaba_ssm import Mamba

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DiffusionModelConfig:
    in_channels: int = 3
    cond_channels: int = 7
    hidden_channels: int = 128
    d_state: int = 128
    d_conv : int = 16
    expand : int = 2

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:

        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = time[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)