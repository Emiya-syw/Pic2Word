import torch
from torch import nn


class SingleQueryQFormerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.text_ln = nn.LayerNorm(dim)
        self.text_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_drop = nn.Dropout(dropout)

        self.image_ln = nn.LayerNorm(dim)
        self.image_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_drop = nn.Dropout(dropout)

        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, image_tokens, text_tokens, text_mask=None):
        text_key_padding_mask = None
        if text_mask is not None:
            text_key_padding_mask = ~text_mask.bool()

        q = q + self.text_drop(
            self.text_attn(
                query=self.text_ln(q),
                key=text_tokens,
                value=text_tokens,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )[0]
        )
        q = q + self.image_drop(
            self.image_attn(
                query=self.image_ln(q),
                key=image_tokens,
                value=image_tokens,
                need_weights=False,
            )[0]
        )
        q = q + self.ffn(self.ffn_ln(q))
        return q


class SingleQueryQFormer(nn.Module):
    """
    Lightweight single-query Q-Former.

    Inputs:
        image_tokens: [B, N_v, D_v]
        text_tokens:  [B, N_t, D_t]
        text_mask:    [B, N_t] (1=valid token)

    Output:
        z0: [B, D]
    """

    def __init__(
        self,
        dim,
        image_dim=None,
        text_dim=None,
        num_layers=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        query_init_std=0.02,
        use_input_proj=True,
    ):
        super().__init__()
        image_dim = image_dim or dim
        text_dim = text_dim or dim

        self.image_proj = nn.Identity()
        self.text_proj = nn.Identity()
        if use_input_proj or image_dim != dim:
            self.image_proj = nn.Linear(image_dim, dim)
        if use_input_proj or text_dim != dim:
            self.text_proj = nn.Linear(text_dim, dim)

        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.query, std=query_init_std)

        self.layers = nn.ModuleList(
            [
                SingleQueryQFormerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_ln = nn.LayerNorm(dim)

    def forward(self, image_tokens, text_tokens, text_mask=None):
        if image_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("image_tokens and text_tokens must be [B, N, D].")

        batch_size = image_tokens.size(0)
        q = self.query.expand(batch_size, -1, -1)

        image_tokens = self.image_proj(image_tokens)
        text_tokens = self.text_proj(text_tokens)

        for layer in self.layers:
            q = layer(q, image_tokens=image_tokens, text_tokens=text_tokens, text_mask=text_mask)

        q = self.out_ln(q)
        return q[:, 0, :]
