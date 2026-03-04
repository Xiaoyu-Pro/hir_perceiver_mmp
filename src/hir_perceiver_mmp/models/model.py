from typing import Dict, Tuple

import math
import torch
import torch.nn as nn

from config import get_default_config

from .fusion import HiRPerceiverFusion
from .heads import ClassificationHead, PretrainHeads
from .reasoner import SystemCausalReasoner


class HiRPerceiverMMPModel(nn.Module):
    """封装 Tokenizers + HiR-Perceiver Fusion + System Causal Reasoner + Heads。"""

    def __init__(
        self,
        metric_dim: int,
        log_dim: int,
        trace_dim: int,
    ) -> None:
        super().__init__()
        cfg = get_default_config()
        d_model = cfg.backbone.d_model

        self.fusion = HiRPerceiverFusion(
            metric_dim=metric_dim,
            log_dim=log_dim,
            trace_dim=trace_dim,
            d_model=d_model,
            n_heads=cfg.backbone.n_heads,
            local_latent_len=cfg.backbone.local_latent_len,
            global_latent_len=cfg.backbone.global_latent_len,
            global_self_layers=cfg.backbone.global_self_layers,
            dropout=cfg.backbone.dropout,
            use_reliability_gate=cfg.backbone.use_reliability_gate,
            use_trace_bias=cfg.backbone.use_trace_bias,
        )

        self.reasoner_enabled = cfg.reasoner.enabled
        if self.reasoner_enabled:
            self.reasoner = SystemCausalReasoner(
                d_model=d_model,
                n_heads=cfg.reasoner.n_heads,
                depth=cfg.reasoner.depth,
                dropout=cfg.reasoner.dropout,
            )
        else:
            self.reasoner = None  # type: ignore[assignment]

        self.cls_head = ClassificationHead(d_model=d_model, hidden_dim=64, dropout=0.1)

        # 预训练重建头：metric_flat/log_vec/trace_vec
        metric_flat_dim = cfg.data.metric_max_t * metric_dim
        self.pretrain_heads = PretrainHeads(
            d_model=d_model,
            metric_out_dim=metric_flat_dim,
            log_out_dim=log_dim,
            trace_out_dim=trace_dim,
            hidden_dim=128,
        )

    # --- API ---

    def forward_backbone(self, metric: torch.Tensor, log_vec: torch.Tensor, trace_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z_global, aux = self.fusion(metric, log_vec, trace_vec)
        return z_global, aux

    def forward_reasoner(self, z_global: torch.Tensor, aux: Dict) -> torch.Tensor:  # aux 为对齐接口
        if self.reasoner_enabled and self.reasoner is not None:
            return self.reasoner(z_global)
        return z_global

    def forward_cls(self, metric: torch.Tensor, log_vec: torch.Tensor, trace_vec: torch.Tensor) -> torch.Tensor:
        z_global, aux = self.forward_backbone(metric, log_vec, trace_vec)
        z = self.forward_reasoner(z_global, aux)
        logits = self.cls_head(z)
        return logits

    def forward_pretrain(self, metric: torch.Tensor, log_vec: torch.Tensor, trace_vec: torch.Tensor):
        """用于 C3-MMP 预训练。返回 (metric_rec, log_rec, trace_rec, z_global)。"""

        z_global, aux = self.forward_backbone(metric, log_vec, trace_vec)
        metric_rec, log_rec, trace_rec = self.pretrain_heads(z_global)
        return metric_rec, log_rec, trace_rec, z_global

    def reset_cls_head(self) -> None:
        """在微调前重置分类头参数。"""
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_((m.weight), a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
