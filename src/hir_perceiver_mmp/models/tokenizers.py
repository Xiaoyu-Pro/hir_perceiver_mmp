import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class MetricTokenizer(nn.Module):
    """Metric Tokenizer: [B, Tmax, F] -> [B, Tmax, d_model]"""

    def __init__(self, in_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,F]
        return self.proj(x)


class VectorTokenizer(nn.Module):
    """向量视作 token：value-gated embedding。

    输入: [B, D]
    输出: [B, D, d_model]
    """

    def __init__(self, dim: int, d_model: int) -> None:
        super().__init__()
        self.dim = dim
        self.d_model = d_model
        self.embeddings = nn.Parameter(torch.randn(dim, d_model) / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        # embeddings: [D, d_model]
        # 输出: [B, D, d_model]
        return x.unsqueeze(-1) * self.embeddings.unsqueeze(0)


class LogTokenizer(VectorTokenizer):
    """Log Tokenizer: [B, D_log] -> [B, D_log, d_model]"""

    pass


class TraceTokenizer(VectorTokenizer):
    """Trace Tokenizer: [B, D_trace] -> [B, D_trace, d_model]"""

    pass
