from typing import Tuple

import torch
import torch.nn as nn


class SystemCausalReasoner(nn.Module):
    """System Causal Reasoner：对 z_global 做 state/behavior/path 分区 + K 层循环推理。"""

    def __init__(self, d_model: int, n_heads: int = 4, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.depth = depth

        self.state_proj = nn.Linear(d_model, d_model)
        self.behavior_proj = nn.Linear(d_model, d_model)
        self.path_proj = nn.Linear(d_model, d_model)

        self.attn_sb = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_bp = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_ps = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln_state = nn.LayerNorm(d_model)
        self.ln_behavior = nn.LayerNorm(d_model)
        self.ln_path = nn.LayerNorm(d_model)

        self.fuse_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, z_global: torch.Tensor) -> torch.Tensor:
        """输入 z_global[B,d]，输出 z_reasoned[B,d]。"""

        B = z_global.size(0)
        z_state = self.state_proj(z_global).unsqueeze(1)  # [B,1,d]
        z_behavior = self.behavior_proj(z_global).unsqueeze(1)
        z_path = self.path_proj(z_global).unsqueeze(1)

        for _ in range(self.depth):
            # state <- behavior
            updated_state, _ = self.attn_sb(z_state, z_behavior, z_behavior)
            z_state = self.ln_state(z_state + updated_state)

            # behavior <- path
            updated_behavior, _ = self.attn_bp(z_behavior, z_path, z_path)
            z_behavior = self.ln_behavior(z_behavior + updated_behavior)

            # path <- state
            updated_path, _ = self.attn_ps(z_path, z_state, z_state)
            z_path = self.ln_path(z_path + updated_path)

        concat = torch.cat([z_state, z_behavior, z_path], dim=1).reshape(B, -1)
        z_reasoned = self.fuse_mlp(concat)
        return z_reasoned
