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
        use_metric_log1p: bool = False,
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
        self.use_metric_log1p = use_metric_log1p
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

        if self.use_metric_log1p:
            metric = np.log1p(np.maximum(metric, 0.0))

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


def _compute_metric_stats(metric_data: Dict[str, np.ndarray], window_ids: Iterable[str], use_metric_log1p: bool) -> NormalizationStats:
    arrays = []
    for win_id in window_ids:
        arr = metric_data[win_id]
        if use_metric_log1p:
            arr = np.log1p(np.maximum(arr, 0.0))
        arrays.append(arr)
    concat = np.concatenate([a.reshape(-1, a.shape[1]) for a in arrays], axis=0)
    mean = concat.mean(axis=0, keepdims=True)
    std = concat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return NormalizationStats(metric_mean=mean.astype(np.float32), metric_std=std.astype(np.float32))


def _split_ids(
    all_ids: List[str],
    labels: Dict[str, int],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """按标签做简单分层划分，尽量保证 train/val/test 都包含一定数量的正样本。"""

    ids = list(all_ids)
    rng = random.Random(seed)

    pos_ids = [i for i in ids if labels.get(i, 0) == 1]
    neg_ids = [i for i in ids if labels.get(i, 0) == 0]

    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    def _split_group(group_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(group_ids)
        if n == 0:
            return [], [], []
        if n == 1:
            return group_ids, [], []
        if n == 2:
            return [group_ids[0]], [group_ids[1]], []

        # n >= 3，按比例分配并保证三份都非空
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        n_test = n - n_train - n_val

        if n_train < 1:
            n_train = 1
        if n_val < 1:
            n_val = 1
        if n_test < 1:
            n_test = 1

        total = n_train + n_val + n_test
        while total > n:
            # 从样本数最多的那一份减去 1
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            total = n_train + n_val + n_test

        train = group_ids[:n_train]
        val = group_ids[n_train : n_train + n_val]
        test = group_ids[n_train + n_val : n_train + n_val + n_test]
        return train, val, test

    pos_train, pos_val, pos_test = _split_group(pos_ids)
    neg_train, neg_val, neg_test = _split_group(neg_ids)

    train_ids = pos_train + neg_train
    val_ids = pos_val + neg_val
    test_ids = pos_test + neg_test

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return train_ids, val_ids, test_ids


def load_datasets_from_dir(data_dir: str, metric_max_t: int, ratios: Tuple[float, float, float], split_seed: int):
    """从目录中加载对齐后的 train/val/test 数据集及维度信息。

    data_dir 参数用于示例数据模式（cfg.data.mode="sample"），真实数据模式下会根据
    cfg.data.real_root_dir + dataset_name + time_range + window_type + service_name 重新拼接。
    """

    cfg = get_default_config()
    data_cfg = cfg.data

    if data_cfg.mode == "real":
        # 根据真实数据配置拼接最终目录
        base_dir = data_cfg.real_root_dir
        final_dir = os.path.join(
            base_dir,
            data_cfg.dataset_name,
            data_cfg.time_range,
            data_cfg.window_type,
            data_cfg.service_name,
        )
        metric_file = data_cfg.metric_filename
        log_file = data_cfg.log_filename
        trace_file = data_cfg.trace_filename
        label_file = data_cfg.label_filename
    else:
        # 示例数据模式，沿用原有相对目录和默认文件名
        final_dir = data_dir
        metric_file = "metric.json"
        log_file = "log.json"
        trace_file = "trace.json"
        label_file = "label.json"

    print(
        "[DataConfig] mode={mode}, base_dir={base}, dataset={dataset}, time_range={tr}, window_type={wt}, "
        "service={svc}, data_dir={dd}, metric_file={mf}, log_file={lf}, trace_file={tf}, label_file={lab}".format(
            mode=data_cfg.mode,
            base=data_cfg.real_root_dir,
            dataset=data_cfg.dataset_name,
            tr=data_cfg.time_range,
            wt=data_cfg.window_type,
            svc=data_cfg.service_name,
            dd=final_dir,
            mf=metric_file,
            lf=log_file,
            tf=trace_file,
            lab=label_file,
        )
    )

    metric_json = _load_json(os.path.join(final_dir, metric_file))
    log_json = _load_json(os.path.join(final_dir, log_file))
    trace_json = _load_json(os.path.join(final_dir, trace_file))
    label_json = _load_json(os.path.join(final_dir, label_file))

    metric_keys = _infer_metric_keys(metric_json)
    metric_data = _build_metric_arrays(metric_json, metric_keys)
    log_data, d_log = _build_log_arrays(log_json)
    trace_data, d_trace = _build_trace_arrays(trace_json)
    labels = _build_labels(label_json)

    # 按 window_id 交集对齐
    id_sets = [set(metric_data.keys()), set(log_data.keys()), set(trace_data.keys()), set(labels.keys())]
    inter_ids = sorted(set.intersection(*id_sets))

    train_ids, val_ids, test_ids = _split_ids(inter_ids, labels, ratios, split_seed)

    if data_cfg.share_val_and_test:
        test_ids = list(val_ids)
        print("[DataSplit] share_val_and_test=True, TEST 将与 VAL 使用相同的窗口集合")

    norm_stats = _compute_metric_stats(metric_data, train_ids, data_cfg.use_metric_log1p)

    def _stats(name: str, ids: List[str]) -> None:
        n = len(ids)
        pos = sum(labels[i] == 1 for i in ids)
        neg = n - pos
        print(f"[DataSplit] {name}: windows={n} pos={pos} neg={neg}")

    total_windows = len(inter_ids)
    total_pos = sum(labels[i] == 1 for i in inter_ids)
    total_neg = total_windows - total_pos
    print(f"[DataStats] total_windows={total_windows} pos={total_pos} neg={total_neg}")
    _stats("TRAIN", train_ids)
    _stats("VAL", val_ids)
    _stats("TEST", test_ids)

    use_metric_zscore = data_cfg.use_metric_zscore
    use_metric_log1p = data_cfg.use_metric_log1p
    use_log_log1p = data_cfg.use_log_log1p
    use_trace_log1p = data_cfg.use_trace_log1p

    train_ds = MMPDataset(metric_data, log_data, trace_data, labels, train_ids, metric_max_t, norm_stats, use_metric_zscore, use_metric_log1p, use_log_log1p, use_trace_log1p)
    val_ds = MMPDataset(metric_data, log_data, trace_data, labels, val_ids, metric_max_t, norm_stats, use_metric_zscore, use_metric_log1p, use_log_log1p, use_trace_log1p)
    test_ds = MMPDataset(metric_data, log_data, trace_data, labels, test_ids, metric_max_t, norm_stats, use_metric_zscore, use_metric_log1p, use_log_log1p, use_trace_log1p)

    num_features = train_ds.num_features

    feature_dims = {
        "metric_in": num_features,
        "log_in": d_log,
        "trace_in": d_trace,
    }

    return train_ds, val_ds, test_ds, feature_dims, norm_stats
