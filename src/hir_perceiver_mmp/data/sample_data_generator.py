import argparse
import json
import os
import random
from typing import List

import numpy as np


def generate_metric(num_windows: int, metric_keys: List[str], t_min: int, t_max: int):
    data = {}
    for i in range(num_windows):
        win_id = str(i)
        length = random.randint(t_min, t_max)
        records = []
        base = random.random() * 0.5
        for t in range(length):
            record = {"timestamp": t}
            for k in metric_keys:
                # 正常样本：平滑波动；异常样本：在后半段抬升
                value = base + 0.05 * np.sin(t / 3.0)
                if i % 5 == 0 and t > length // 2:
                    value += 0.5
                record[k] = float(value + 0.02 * np.random.randn())
            records.append(record)
        data[win_id] = {
            "time_range": f"2021-07-01 00:00:00 - 2021-07-01 00:00:{length:02d}",
            "data": records,
        }
    return data


def generate_log(num_windows: int, num_event_ids: int, levels: List[str]):
    metadata = {
        "sorted_eventids": [f"e{i}" for i in range(num_event_ids)],
        "sorted_levels": levels,
    }
    data = {}
    for i in range(num_windows):
        win_id = str(i)
        counts = np.random.poisson(2.0, size=num_event_ids).astype(float)
        level_counts = np.random.poisson(1.0, size=len(levels)).astype(float)
        total = float(counts.sum() + level_counts.sum())
        # 异常窗口增加 ERROR 级别
        if i % 5 == 0:
            level_counts[-1] += 10.0
            total += 10.0
        data[win_id] = {
            "eventid_count": counts.tolist(),
            "levels": level_counts.tolist(),
            "total_count": total,
        }
    return {"metadata": metadata, "data": data}


def generate_trace(num_windows: int, num_spans: int):
    metadata = {"sorted_names": [f"span{i}" for i in range(num_spans)]}
    data = {}
    for i in range(num_windows):
        win_id = str(i)
        counts = np.random.poisson(1.0, size=num_spans).astype(float)
        durations = np.random.exponential(scale=0.1, size=num_spans).astype(float)
        total_count = float(counts.sum())
        total_duration = float(durations.sum())
        # 异常窗口增加部分 span 时长
        if i % 5 == 0:
            durations += 0.3
            total_duration = float(durations.sum())
        data[win_id] = {
            "counts": counts.tolist(),
            "durations": durations.tolist(),
            "total_count": total_count,
            "total_duration": total_duration,
        }
    return {"metadata": metadata, "data": data}


def generate_labels(num_windows: int):
    data = {}
    for i in range(num_windows):
        win_id = str(i)
        # 每 5 个窗口一个异常
        label = (i % 5 == 0)
        data[win_id] = {"time_range": "-", "label": bool(label)}
    return data


def main():
    parser = argparse.ArgumentParser(description="生成最小示例多模态数据集")
    parser.add_argument("--output_dir", type=str, default="data/sample_run")
    parser.add_argument("--num_windows", type=int, default=64)
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)

    os.makedirs(args.output_dir, exist_ok=True)

    metric = generate_metric(num_windows=args.num_windows, metric_keys=["kpiA", "kpiB", "kpiC", "kpiD"], t_min=6, t_max=12)
    log = generate_log(num_windows=args.num_windows, num_event_ids=6, levels=["INFO", "WARN", "ERROR"])
    trace = generate_trace(num_windows=args.num_windows, num_spans=5)
    label = generate_labels(num_windows=args.num_windows)

    with open(os.path.join(args.output_dir, "metric.json"), "w", encoding="utf-8") as f:
        json.dump(metric, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "label.json"), "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)

    print(f"示例数据已生成到: {args.output_dir}")


if __name__ == "__main__":
    main()
