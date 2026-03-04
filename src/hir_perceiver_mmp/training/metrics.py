from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 200) -> Dict[str, float]:
    thresholds = np.linspace(0.0, 1.0, num=num_thresholds + 1)
    best_f1 = -1.0
    best_thr = 0.5
    best_p = 0.0
    best_r = 0.0
    best_tp = 0.0
    best_fp = 0.0
    best_tn = 0.0
    best_fn = 0.0
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        tn = float(((y_pred == 0) & (y_true == 0)).sum())
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_p = precision
            best_r = recall
            best_tp = tp
            best_fp = fp
            best_tn = tn
            best_fn = fn
    return {
        "best_threshold": best_thr,
        "precision": best_p,
        "recall": best_r,
        "f1": best_f1,
        "tp": best_tp,
        "fp": best_fp,
        "tn": best_tn,
        "fn": best_fn,
    }


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, fixed_threshold: Optional[float] = None) -> Dict[str, float]:
    roc = compute_roc_auc(y_true, y_score)
    pr = compute_pr_auc(y_true, y_score)

    if fixed_threshold is None:
        thr_stats = find_best_threshold(y_true, y_score)
        threshold = thr_stats["best_threshold"]
        precision = thr_stats["precision"]
        recall = thr_stats["recall"]
        f1 = thr_stats["f1"]
        tp = thr_stats["tp"]
        fp = thr_stats["fp"]
        tn = thr_stats["tn"]
        fn = thr_stats["fn"]
    else:
        # 使用给定阈值计算混淆矩阵及指标
        threshold = fixed_threshold
        y_pred = (y_score >= threshold).astype(int)
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        tn = float(((y_pred == 0) & (y_true == 0)).sum())
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "best_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
