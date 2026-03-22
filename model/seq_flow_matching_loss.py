import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TokenFlowLossOutput:
    loss: torch.Tensor
    loss_dict: Dict[str, torch.Tensor]
    aux: Dict[str, torch.Tensor]


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    pred/target: [B, L, D]
    mask: [B, L], 1 for valid
    """
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [B, L]

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    return loss.mean()


def masked_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    1 - cosine similarity
    pred/target: [B, L, D]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    loss = 1.0 - (pred * target).sum(dim=-1)  # [B, L]

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    return loss.mean()


def build_unchanged_mask_from_cosine(
    src_tokens: torch.Tensor,
    tgt_tokens: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.98,
) -> torch.Tensor:
    """
    用 src/tgt 同位置 token 的 cosine 判断该位置是否“基本不变”
    return: [B, L] bool
    """
    src_n = F.normalize(src_tokens, dim=-1)
    tgt_n = F.normalize(tgt_tokens, dim=-1)
    sim = (src_n * tgt_n).sum(dim=-1)  # [B, L]
    unchanged = sim > threshold

    if src_mask is not None:
        unchanged = unchanged & src_mask.bool()

    return unchanged


class TokenFlowMatchingLoss(nn.Module):
    """
    只负责：
    - FM loss
    - endpoint loss
    - token cosine alignment loss
    - keep/preservation loss
    - optional direct endpoint loss

    输入：
    - x0: source text tokens
    - x1: target text tokens
    - x_t: interpolated tokens at time t
    - velocity: model predicted velocity at x_t
    - t: sampled time
    """

    def __init__(
        self,
        lambda_fm: float = 1.0,
        lambda_end: float = 1.0,
        lambda_tok: float = 0.2,
        lambda_keep: float = 0.5,
        lambda_direct_end: float = 0.0,
        keep_threshold: float = 0.98,
    ):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_end = lambda_end
        self.lambda_tok = lambda_tok
        self.lambda_keep = lambda_keep
        self.lambda_direct_end = lambda_direct_end
        self.keep_threshold = keep_threshold

    def build_ut(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        linear path target velocity:
        u_t = x1 - x0
        """
        return x1 - x0

    def predict_x1(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        一步 Euler 从 t 积分到 1:
        x1_hat = x_t + (1 - t) * v
        """
        if t.dim() == 1:
            remain = (1.0 - t)[:, None, None]     # [B,1,1]
        elif t.dim() == 2 and t.shape[-1] == 1:
            remain = (1.0 - t)[:, None, :]        # [B,1,1]
        else:
            raise ValueError(f"Unsupported t shape: {t.shape}")

        return x_t + remain * velocity

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        t: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        pred_x1_direct: Optional[torch.Tensor] = None,
    ) -> TokenFlowLossOutput:
        """
        x0, x1, x_t, velocity: [B, L, D]
        t: [B] or [B,1]
        token_mask: [B, L]
        pred_x1_direct: optional [B, L, D]
        """
        device = x0.device
        dtype = x0.dtype

        # ground-truth velocity
        u_t = self.build_ut(x0, x1)

        # euler endpoint
        x1_hat = self.predict_x1(x_t, velocity, t)

        # 1) FM
        loss_fm = masked_mse(velocity, u_t, token_mask)

        # 2) endpoint reconstruction
        loss_end = masked_mse(x1_hat, x1, token_mask)

        # 3) token cosine alignment
        loss_tok = masked_cosine_loss(x1_hat, x1, token_mask)

        # 4) keep loss on unchanged positions
        unchanged_mask = build_unchanged_mask_from_cosine(
            src_tokens=x0,
            tgt_tokens=x1,
            src_mask=token_mask,
            threshold=self.keep_threshold,
        )

        if unchanged_mask.any():
            loss_keep = masked_mse(x1_hat, x0, unchanged_mask)
        else:
            loss_keep = torch.zeros([], device=device, dtype=dtype)

        # 5) optional direct endpoint prediction
        if pred_x1_direct is not None and self.lambda_direct_end > 0:
            loss_direct_end = masked_mse(pred_x1_direct, x1, token_mask)
        else:
            loss_direct_end = torch.zeros([], device=device, dtype=dtype)

        total_loss = (
            self.lambda_fm * loss_fm
            + self.lambda_end * loss_end
            + self.lambda_tok * loss_tok
            + self.lambda_keep * loss_keep
            + self.lambda_direct_end * loss_direct_end
        )

        loss_dict = {
            "loss": total_loss.detach(),
            "fm": loss_fm.detach(),
            "end": loss_end.detach(),
            "tok": loss_tok.detach(),
            "keep": loss_keep.detach(),
            "direct_end": loss_direct_end.detach(),
        }

        aux = {
            "u_t": u_t.detach(),
            "x1_hat": x1_hat.detach(),
            "unchanged_mask": unchanged_mask.detach(),
        }

        return TokenFlowLossOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            aux=aux,
        )