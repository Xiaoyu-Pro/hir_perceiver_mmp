import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import get_default_config


@dataclass
class NormalizationStats:
    metric_mean: np.ndarray
    metric_std: np.ndarray


class MMPDataset(Dataset):
    """多模态窗口级数据集。

    __getitem__ 返回：metric[Tmax,F]、log_vec[D_log]、trace_vec[D_trace]、label[]、key
    """

    def __init__(
        self,
        metric_data: Dict[str, np.ndarray],
        log_data: Dict[str, np.ndarray],
        trace_data: Dict[str, np.ndarray],
        labels: Dict[str, int],
        window_ids: List[str],
        metric_max_t: int,
        norm_stats: NormalizationStats,
        use_metric_zscore: bool = True,
        use_log_log1p: bool = True,
        use_trace_log1p: bool = True,
    ) -> None:
        self.metric_data = metric_data
        self.log_data = log_data
        self.trace_data = trace_data
        self.labels = labels
        self.window_ids = window_ids
        self.metric_max_t = metric_max_t
        self.norm_stats = norm_stats
        self.use_metric_zscore = use_metric_zscore
        self.use_log_log1p = use_log_log1p
        self.use_trace_log1p = use_trace_log1p

        # 推断维度
        any_metric = next(iter(metric_data.values()))
        self.num_features = any_metric.shape[1]
        any_log = next(iter(log_data.values()))
        any_trace = next(iter(trace_data.values()))
        self.d_log = any_log.shape[0]
        self.d_trace = any_trace.shape[0]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.window_ids)

    def _pad_or_truncate_metric(self, arr: np.ndarray) -> np.ndarray:
        T, F = arr.shape
        Tmax = self.metric_max_t
        if T >= Tmax:
            return arr[:Tmax]
        pad_len = Tmax - T
        pad = np.zeros((pad_len, F), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def __getitem__(self, idx: int):  # type: ignore[override]
        key = self.window_ids[idx]
        metric = self.metric_data[key].astype(np.float32)
        log_vec = self.log_data[key].astype(np.float32)
        trace_vec = self.trace_data[key].astype(np.float32)
        label = int(self.labels[key])

        metric = self._pad_or_truncate_metric(metric)

        if self.use_metric_zscore:
            metric = (metric - self.norm_stats.metric_mean) / self.norm_stats.metric_std

        if self.use_log_log1p:
            log_vec = np.log1p(np.maximum(log_vec, 0.0))
        if self.use_trace_log1p:
            trace_vec = np.log1p(np.maximum(trace_vec, 0.0))

        metric_t = torch.from_numpy(metric)
        log_t = torch.from_numpy(log_vec)
        trace_t = torch.from_numpy(trace_vec)
        label_t = torch.tensor(label, dtype=torch.long)

        return metric_t, log_t, trace_t, label_t, key


def collate_fn(batch):
    metrics, logs, traces, labels, keys = zip(*batch)
    metric_batch = torch.stack(metrics, dim=0)
    log_batch = torch.stack(logs, dim=0)
    trace_batch = torch.stack(traces, dim=0)
    label_batch = torch.stack(labels, dim=0)
    return metric_batch, log_batch, trace_batch, label_batch, list(keys)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_metric_keys(metric_json: Dict[str, dict]) -> List[str]:
    first_key = sorted(metric_json.keys())[0]
    first_record = metric_json[first_key]["data"][0]
    keys = [k for k in first_record.keys() if k != "timestamp"]
    keys.sort()
    return keys


def _build_metric_arrays(metric_json: Dict[str, dict], metric_keys: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for win_id, content in metric_json.items():
        records = content["data"]
        arr = np.zeros((len(records), len(metric_keys)), dtype=np.float32)
        for t, rec in enumerate(records):
            for j, k in enumerate(metric_keys):
                arr[t, j] = float(rec.get(k, 0.0))
        out[win_id] = arr
    return out


def _build_log_arrays(log_json: Dict[str, dict]) -> Tuple[Dict[str, np.ndarray], int]:
    metadata = log_json["metadata"]
    data = log_json["data"]
    sorted_eventids: List[str] = metadata["sorted_eventids"]
    sorted_levels: List[str] = metadata["sorted_levels"]
    d_log = 1 + len(sorted_eventids) + len(sorted_levels)
    out: Dict[str, np.ndarray] = {}
    for win_id, rec in data.items():
        vec = np.zeros(d_log, dtype=np.float32)
        total = float(rec.get("total_count", 0.0))
        event_counts = rec.get("eventid_count", [0.0] * len(sorted_eventids))
        level_counts = rec.get("levels", [0.0] * len(sorted_levels))
        vec[0] = total
        vec[1 : 1 + len(sorted_eventids)] = np.array(event_counts, dtype=np.float32)
        start = 1 + len(sorted_eventids)
        vec[start : start + len(sorted_levels)] = np.array(level_counts, dtype=np.float32)
        out[win_id] = vec
    return out, d_log


def _build_trace_arrays(trace_json: Dict[str, dict]) -> Tuple[Dict[str, np.ndarray], int]:
    metadata = trace_json["metadata"]
    data = trace_json["data"]
    sorted_names: List[str] = metadata["sorted_names"]
    d_trace = 2 + 2 * len(sorted_names)
    out: Dict[str, np.ndarray] = {}
    for win_id, rec in data.items():
        vec = np.zeros(d_trace, dtype=np.float32)
        total_count = float(rec.get("total_count", 0.0))
        total_duration = float(rec.get("total_duration", 0.0))
        counts = rec.get("counts", [0.0] * len(sorted_names))
        durations = rec.get("durations", [0.0] * len(sorted_names))
        vec[0] = total_count
        vec[1] = total_duration
        vec[2 : 2 + len(sorted_names)] = np.array(counts, dtype=np.float32)
        start = 2 + len(sorted_names)
        vec[start : start + len(sorted_names)] = np.array(durations, dtype=np.float32)
        out[win_id] = vec
    return out, d_trace


def _build_labels(label_json: Dict[str, dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for win_id, rec in label_json.items():
        label_bool = bool(rec.get("label", False))
        out[win_id] = int(label_bool)
    return out


def _compute_metric_stats(metric_data: Dict[str, np.ndarray], window_ids: Iterable[str]) -> NormalizationStats:
    arrays = []
    for win_id in window_ids:
        arrays.append(metric_data[win_id])
    concat = np.concatenate([a.reshape(-1, a.shape[1]) for a in arrays], axis=0)
    mean = concat.mean(axis=0, keepdims=True)
    std = concat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return NormalizationStats(metric_mean=mean.astype(np.float32), metric_std=std.astype(np.float32))


def _split_ids(all_ids: List[str], ratios: Tuple[float, float, float], seed: int) -> Tuple[List[str], List[str], List[str]]:
    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    if not test_ids:
        test_ids = val_ids
    return train_ids, val_ids, test_ids


def load_datasets_from_dir(data_dir: str, metric_max_t: int, ratios: Tuple[float, float, float], split_seed: int):
    """从目录中加载对齐后的 train/val/test 数据集及维度信息。"""

    metric_json = _load_json(os.path.join(data_dir, "metric.json"))
    log_json = _load_json(os.path.join(data_dir, "log.json"))
    trace_json = _load_json(os.path.join(data_dir, "trace.json"))
    label_json = _load_json(os.path.join(data_dir, "label.json"))

    metric_keys = _infer_metric_keys(metric_json)
    metric_data = _build_metric_arrays(metric_json, metric_keys)
    log_data, d_log = _build_log_arrays(log_json)
    trace_data, d_trace = _build_trace_arrays(trace_json)
    labels = _build_labels(label_json)

    # 按 window_id 交集对齐
    id_sets = [set(metric_data.keys()), set(log_data.keys()), set(trace_data.keys()), set(labels.keys())]
    inter_ids = sorted(set.intersection(*id_sets))

    train_ids, val_ids, test_ids = _split_ids(inter_ids, ratios, split_seed)

    norm_stats = _compute_metric_stats(metric_data, train_ids)

    cfg = get_default_config()
    use_metric_zscore = cfg.data.use_metric_zscore
    use_log_log1p = cfg.data.use_log_log1p
    use_trace_log1p = cfg.data.use_trace_log1p

    train_ds = MMPDataset(metric_data, log_data, trace_data, labels, train_ids, metric_max_t, norm_stats, use_metric_zscore, use_log_log1p, use_trace_log1p)
    val_ds = MMPDataset(metric_data, log_data, trace_data, labels, val_ids, metric_max_t, norm_stats, use_metric_zscore, use_log_log1p, use_trace_log1p)
    test_ds = MMPDataset(metric_data, log_data, trace_data, labels, test_ids, metric_max_t, norm_stats, use_metric_zscore, use_log_log1p, use_trace_log1p)

    num_features = train_ds.num_features

    feature_dims = {
        "metric_in": num_features,
        "log_in": d_log,
        "trace_in": d_trace,
    }

    return train_ds, val_ds, test_ds, feature_dims, norm_stats
