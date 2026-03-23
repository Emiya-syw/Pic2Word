import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B, 1]
        half = self.dim // 2
        device = t.device
        scale = math.log(10000) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=device) * -scale)
        x = t * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, expansion=2):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(cond_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.gate = nn.Linear(cond_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + torch.sigmoid(self.gate(cond)) * h


class ConditionalFlowNet(nn.Module):
    def __init__(self, dim, time_dim=128, hidden_dim=4096, depth=3):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.x_proj = nn.Linear(dim, hidden_dim)
        self.delta_proj = nn.Linear(dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x_t, delta, e_m, t):
        # x_t, delta, e_m: [B, D]
        # t: [B, 1]
        t_emb = self.time_embed(t)
        h = self.x_proj(x_t) + self.delta_proj(delta)
        h = self.input_norm(h)
        cond = self.cond_proj(torch.cat([e_m, t_emb], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(self.out_norm(h))
