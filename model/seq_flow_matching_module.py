import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# utility
# =========================

def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int):
    """
    x: [B, L, D]
    mask: [B, L] with 1 for valid, 0 for pad
    """
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.float()
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] or [B, 1], in [0,1]
    return: [B, dim]
    """
    if t.dim() == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def build_unchanged_mask_from_cosine(
    src_tokens: torch.Tensor,
    tgt_tokens: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.98,
) -> torch.Tensor:
    """
    粗糙地找“不变 token”位置：
    用 src/tgt 同位置 token 的 cosine 相似度判断。
    返回 [B, L] bool
    """
    src_n = F.normalize(src_tokens, dim=-1)
    tgt_n = F.normalize(tgt_tokens, dim=-1)
    sim = (src_n * tgt_n).sum(dim=-1)  # [B, L]
    unchanged = sim > threshold
    if src_mask is not None:
        unchanged = unchanged & src_mask.bool()
    return unchanged


# =========================
# modules
# =========================

class TimeConditionedLayerNorm(nn.Module):
    """
    AdaLN-lite:
    y = LN(x) * (1 + scale(t)) + shift(t)
    """
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        t_emb: [B, time_dim]
        """
        scale_shift = self.to_scale_shift(t_emb)  # [B, 2D]
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q: [B, Lq, D]
        k/v: [B, Lk, D]
        q_mask: [B, Lq] (unused for attention score, but kept for interface consistency)
        kv_mask: [B, Lk], 1 valid
        """
        B, Lq, D = q.shape
        Lk = k.shape[1]

        q = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Lq, Lk]

        if kv_mask is not None:
            mask = kv_mask[:, None, None, :].bool()
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, Lq, Hd]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class VisualResampler(nn.Module):
    """
    用 learnable queries 从视觉 token 中压缩出 K 个条件 token
    """
    def __init__(self, vis_dim: int, model_dim: int, num_queries: int = 16, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, model_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, model_dim) * 0.02)
        self.attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, mlp_ratio=4.0, dropout=dropout)

    def forward(self, vis_tokens: torch.Tensor, vis_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        vis_tokens: [B, Lv, Dv]
        return: [B, K, D]
        """
        B = vis_tokens.shape[0]
        v = self.vis_proj(vis_tokens)
        q = self.queries.expand(B, -1, -1)
        q = q + self.attn(q, v, v, kv_mask=vis_mask)
        q = self.norm(q)
        q = q + self.ffn(q)
        return q


class TokenFlowBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, time_dim: int, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_self = TimeConditionedLayerNorm(dim, time_dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)

        self.norm_src = TimeConditionedLayerNorm(dim, time_dim)
        self.src_cross_attn = MultiHeadAttention(dim, num_heads, dropout)

        self.norm_vis = TimeConditionedLayerNorm(dim, time_dim)
        self.vis_cross_attn = MultiHeadAttention(dim, num_heads, dropout)

        self.norm_ffn = TimeConditionedLayerNorm(dim, time_dim)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_tokens: torch.Tensor,
        vis_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        vis_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self-attention on current state x_t
        h = self.norm_self(x, t_emb)
        x = x + self.dropout(self.self_attn(h, h, h, q_mask=x_mask, kv_mask=x_mask))

        # cross-attn to source text x_0
        h = self.norm_src(x, t_emb)
        x = x + self.dropout(self.src_cross_attn(h, src_tokens, src_tokens, q_mask=x_mask, kv_mask=src_mask))

        # cross-attn to visual condition
        h = self.norm_vis(x, t_emb)
        x = x + self.dropout(self.vis_cross_attn(h, vis_tokens, vis_tokens, q_mask=x_mask, kv_mask=vis_mask))

        # FFN
        h = self.norm_ffn(x, t_emb)
        x = x + self.dropout(self.ffn(h))

        return x


@dataclass
class TokenFlowOutput:
    velocity: torch.Tensor
    pred_x1: Optional[torch.Tensor] = None
    aux: Optional[Dict] = None


class TokenFlowNet(nn.Module):
    def __init__(
        self,
        text_dim: int,
        vis_dim: int,
        model_dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
        num_vis_queries: int = 16,
        dropout: float = 0.1,
        predict_residual: bool = False,
    ):
        super().__init__()
        self.text_in = nn.Linear(text_dim, model_dim)
        self.src_in = nn.Linear(text_dim, model_dim)
        self.vis_resampler = VisualResampler(
            vis_dim=vis_dim,
            model_dim=model_dim,
            num_queries=num_vis_queries,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.blocks = nn.ModuleList([
            TokenFlowBlock(
                dim=model_dim,
                num_heads=num_heads,
                time_dim=time_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(model_dim)
        self.vel_head = nn.Linear(model_dim, text_dim)

        # 可选：直接预测 x1 residual
        self.predict_residual = predict_residual
        if predict_residual:
            self.x1_head = nn.Linear(model_dim, text_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        src_tokens: torch.Tensor,
        vis_tokens: torch.Tensor,
        t: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        vis_mask: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ) -> TokenFlowOutput:
        """
        x_t: [B, Lt, Dt]
        src_tokens: [B, Lt, Dt]
        vis_tokens: [B, Lv, Dv]
        t: [B] or [B,1]
        """
        if x_mask is None:
            x_mask = src_mask

        t_emb = sinusoidal_time_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        x = self.text_in(x_t)
        src = self.src_in(src_tokens)
        vis = self.vis_resampler(vis_tokens, vis_mask=vis_mask)
        vis_resampled_mask = torch.ones(
            vis.shape[:2], device=vis.device, dtype=torch.bool
        )

        for blk in self.blocks:
            x = blk(
                x=x,
                src_tokens=src,
                vis_tokens=vis,
                t_emb=t_emb,
                x_mask=x_mask,
                src_mask=src_mask,
                vis_mask=vis_resampled_mask,
            )

        h = self.final_norm(x)
        velocity = self.vel_head(h)

        pred_x1 = None
        if self.predict_residual:
            pred_x1 = src_tokens + self.x1_head(h)

        return TokenFlowOutput(
            velocity=velocity,
            pred_x1=pred_x1,
            aux={"vis_cond": vis}
        )