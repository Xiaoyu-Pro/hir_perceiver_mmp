from typing import Dict, Tuple

import torch
import torch.nn as nn

from .tokenizers import MetricTokenizer, LogTokenizer, TraceTokenizer


class HiRPerceiverFusion(nn.Module):
    """HiR-Perceiver 融合骨干网络。

    - 三模态 local latents + global latents
    - 模态级可靠性门控 + trace bias
    - 若干层 global self-attention
    """

    def __init__(
        self,
        metric_dim: int,
        log_dim: int,
        trace_dim: int,
        d_model: int,
        n_heads: int = 4,
        local_latent_len: int = 8,
        global_latent_len: int = 8,
        global_self_layers: int = 1,
        dropout: float = 0.1,
        use_reliability_gate: bool = True,
        use_trace_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_reliability_gate = use_reliability_gate
        self.use_trace_bias = use_trace_bias

        self.metric_tokenizer = MetricTokenizer(metric_dim, d_model)
        self.log_tokenizer = LogTokenizer(log_dim, d_model)
        self.trace_tokenizer = TraceTokenizer(trace_dim, d_model)

        self.metric_local_latents = nn.Parameter(torch.randn(local_latent_len, d_model))
        self.log_local_latents = nn.Parameter(torch.randn(local_latent_len, d_model))
        self.trace_local_latents = nn.Parameter(torch.randn(local_latent_len, d_model))
        self.global_latents = nn.Parameter(torch.randn(global_latent_len, d_model))

        self.metric_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.log_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.trace_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.metric_local_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.log_local_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.trace_local_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.local_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)

        self.global_self_blocks = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(global_self_layers)]
        )
        self.global_self_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(global_self_layers)])

        # 质量特征 -> gate
        self.metric_gate_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.log_gate_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.trace_gate_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.trace_bias_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

    @staticmethod
    def _metric_quality_features(metric: torch.Tensor) -> torch.Tensor:
        # metric: [B,T,F]
        zeros = (metric == 0).float().mean(dim=(1, 2), keepdim=True)
        var = metric.var(dim=(1, 2), keepdim=True)
        return torch.cat([zeros, var], dim=-1).squeeze(1)

    @staticmethod
    def _vector_quality_features(vec: torch.Tensor) -> torch.Tensor:
        # vec: [B,D]
        zeros = (vec == 0).float().mean(dim=1, keepdim=True)
        abs_vec = vec.abs()
        sums = abs_vec.sum(dim=1, keepdim=True) + 1e-6
        top1 = abs_vec.max(dim=1, keepdim=True).values / sums
        return torch.cat([zeros, top1], dim=-1)

    @staticmethod
    def _trace_quality_features(vec: torch.Tensor) -> torch.Tensor:
        # vec: [B,D]，前两个维度为 total_count,total_duration
        zeros = (vec == 0).float().mean(dim=1, keepdim=True)
        total_duration = vec[:, 1:2]
        return torch.cat([zeros, total_duration], dim=-1)

    def forward(self, metric: torch.Tensor, log_vec: torch.Tensor, trace_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """前向：返回 z_global 以及辅助信息（质量特征等）。"""

        B = metric.size(0)

        metric_tokens = self.metric_tokenizer(metric)  # [B,T,F] -> [B,T,d]
        log_tokens = self.log_tokenizer(log_vec)  # [B,D_log,d]
        trace_tokens = self.trace_tokenizer(trace_vec)  # [B,D_trace,d]

        # Local latents 读取各自模态 tokens
        def read_local(latents_param, tokens, attn):
            latents = latents_param.unsqueeze(0).expand(B, -1, -1)
            updated, _ = attn(query=latents, key=tokens, value=tokens)
            latents = self.local_norm(latents + updated)
            return latents

        Z_m_local = read_local(self.metric_local_latents, metric_tokens, self.metric_local_attn)
        Z_l_local = read_local(self.log_local_latents, log_tokens, self.log_local_attn)
        Z_t_local = read_local(self.trace_local_latents, trace_tokens, self.trace_local_attn)

        # Global latents 依次读取三模态 local latents
        Z_global = self.global_latents.unsqueeze(0).expand(B, -1, -1)

        q_m = self._metric_quality_features(metric)
        q_l = self._vector_quality_features(log_vec)
        q_t = self._trace_quality_features(trace_vec)

        g_m = torch.sigmoid(self.metric_gate_mlp(q_m)).view(B, 1, 1)
        g_l = torch.sigmoid(self.log_gate_mlp(q_l)).view(B, 1, 1)
        g_t = torch.sigmoid(self.trace_gate_mlp(q_t)).view(B, 1, 1)
        b_t = torch.sigmoid(self.trace_bias_mlp(q_t)).view(B, 1, 1)

        if not self.use_reliability_gate:
            g_m = g_l = g_t = Z_global.new_ones(B, 1, 1)
        if not self.use_trace_bias:
            b_t = Z_global.new_ones(B, 1, 1)

        def read_global(Z_global, local_latents, attn, gate):
            updated, _ = attn(query=Z_global, key=local_latents, value=local_latents)
            Z_global = self.global_norm(Z_global + gate * updated)
            return Z_global

        Z_global = read_global(Z_global, Z_m_local, self.metric_cross_attn, g_m)
        Z_global = read_global(Z_global, Z_l_local, self.log_cross_attn, g_l)
        Z_global = read_global(Z_global, Z_t_local, self.trace_cross_attn, g_t * b_t)

        # Global self-attention blocks
        for attn, ln in zip(self.global_self_blocks, self.global_self_norms):
            updated, _ = attn(Z_global, Z_global, Z_global)
            Z_global = ln(Z_global + updated)

        z_global = Z_global.mean(dim=1)  # [B,d]

        aux = {
            "q_m": q_m,
            "q_l": q_l,
            "q_t": q_t,
        }
        return z_global, aux
