import math,torch
import torch.nn as nn
import torch.nn
from dataclasses import dataclass
import torch.nn.functional as F

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

        # f_i = 1/1000^(i/dim_half - 1) Using this fromula to generate scaling frequencies
        scale = math.log(10000) / (half_dim - 1)

        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = time[:, None].float() * freqs[None, :]

        return torch.cat([args.sin(), args.cos()], dim=-1)

class CondtionalEmbedding(nn.Module):
    def __init__(self,cond_dim: int = 7, emb_dim: int =128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)

class ResBlock1D(nn.Module):

    def __init__(self,  in_channels :int , out_channels: int, emb_dim: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2)
        )

        self.residual - (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:

        res = self.residual(x)

        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.emb_proj(emb).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h) * (1.0 + scale) + shift

        h = self.conv2(F.silu(h))

        return h + res