import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """二分类头：输入 [B,d] 输出 [B] logit。"""

    def __init__(self, d_model: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return logits


class PretrainHeads(nn.Module):
    """预训练重建头：metric/log/trace。"""

    def __init__(self, d_model: int, metric_out_dim: int, log_out_dim: int, trace_out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.metric_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, metric_out_dim),
        )
        self.log_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, log_out_dim),
        )
        self.trace_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trace_out_dim),
        )

    def forward(self, z: torch.Tensor):
        metric_rec = self.metric_head(z)
        log_rec = self.log_head(z)
        trace_rec = self.trace_head(z)
        return metric_rec, log_rec, trace_rec
