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
        freq = torch.exp(torch.arange(half, device=device) * -scale)  # [half]
        x = t * freq.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [B, 2*half]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + x


class ConditionalFlowNet(nn.Module):
    """
    Residual MLP for global-token / vector flow matching.

    Inputs:
        x_t   : [B, D] current state at time t
        delta : [B, D] relative state, usually x_t - x_0
        e_m   : [B, D] modification / condition embedding
        t     : [B, 1] continuous time in [0, 1]

    Output:
        v     : [B, D] predicted velocity
    """
    def __init__(
        self,
        dim,
        time_dim=128,
        hidden_dim=2048,
        num_blocks=4,
        dropout=0.0,
        use_cond_gate=True,
        out_norm=False,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.use_cond_gate = use_cond_gate
        self.out_norm = out_norm

        # x_t + delta + e_m + t_emb
        in_dim = dim * 3 + time_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim=hidden_dim * 4, dropout=dropout)
            for _ in range(num_blocks)
        ])

        if use_cond_gate:
            # delta + e_m + t_emb
            self.gate = nn.Sequential(
                nn.LayerNorm(dim * 2 + time_dim),
                nn.Linear(dim * 2 + time_dim, hidden_dim),
                nn.Sigmoid(),
            )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x_t, delta, e_m, t):
        # x_t, delta, e_m: [B, D]
        # t: [B, 1]

        t_emb = self.time_embed(t)  # [B, time_dim]

        # main input
        h = torch.cat([x_t, delta, e_m, t_emb], dim=-1)  # [B, 3D + T]
        h = self.input_proj(h)  # [B, H]

        # optional condition gate
        if self.use_cond_gate:
            cond = torch.cat([delta, e_m, t_emb], dim=-1)  # [B, 2D + T]
            gate = self.gate(cond)                          # [B, H]
            h = h * gate

        # residual trunk
        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        v = self.out_proj(h)  # [B, D]

        if self.out_norm:
            v = l2norm(v)

        return v