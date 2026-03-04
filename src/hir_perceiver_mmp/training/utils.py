import math
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cpu") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, model_state: dict, extra: Optional[Dict[str, Any]] = None) -> None:
    ckpt = {"model": model_state}
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> dict:
    return torch.load(path, map_location=map_location)


def metric_block_mask(metric: torch.Tensor, ratio: float) -> torch.Tensor:
    """Metric block mask：沿时间维随机置零一段连续区间。"""

    B, T, F = metric.shape
    if ratio <= 0:
        return metric
    block_len = max(1, int(T * ratio))
    out = metric.clone()
    for i in range(B):
        if block_len >= T:
            start = 0
        else:
            start = random.randint(0, T - block_len)
        out[i, start : start + block_len, :] = 0.0
    return out


def log_burst_mask(log_vec: torch.Tensor, ratio: float) -> torch.Tensor:
    """Log burst mask：将向量中 top-k 大的维度置零。"""

    B, D = log_vec.shape
    if ratio <= 0:
        return log_vec
    k = max(1, int(D * ratio))
    out = log_vec.clone()
    abs_vals = out.abs()
    _, indices = torch.topk(abs_vals, k=k, dim=1)
    for i in range(B):
        out[i, indices[i]] = 0.0
    return out


def trace_edge_drop_mask(trace_vec: torch.Tensor, ratio: float) -> torch.Tensor:
    """Trace edge-drop mask：除前 2 维(total_count,total_duration) 外随机按比例置零。"""

    B, D = trace_vec.shape
    if ratio <= 0:
        return trace_vec
    out = trace_vec.clone()
    if D <= 2:
        return out
    span_dim = D - 2
    drop_mask = (torch.rand(B, span_dim, device=trace_vec.device) < ratio).float()
    out[:, 2:] = out[:, 2:] * (1.0 - drop_mask)
    return out


def create_two_masked_views(metric: torch.Tensor, log_vec: torch.Tensor, trace_vec: torch.Tensor, cfg) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """根据 C3-MMP 策略创建两份 masked 视图。"""

    m1 = metric_block_mask(metric, cfg.pretrain.metric_block_mask_ratio)
    m2 = metric_block_mask(metric, cfg.pretrain.metric_block_mask_ratio)

    l1 = log_burst_mask(log_vec, cfg.pretrain.log_burst_mask_ratio)
    l2 = log_burst_mask(log_vec, cfg.pretrain.log_burst_mask_ratio)

    t1 = trace_edge_drop_mask(trace_vec, cfg.pretrain.trace_edge_drop_ratio)
    t2 = trace_edge_drop_mask(trace_vec, cfg.pretrain.trace_edge_drop_ratio)

    return (m1, l1, t1), (m2, l2, t2)
