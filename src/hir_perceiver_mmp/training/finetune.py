import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.hir_perceiver_mmp.data.dataset import collate_fn, load_datasets_from_dir
from src.hir_perceiver_mmp.models.model import HiRPerceiverMMPModel
from src.hir_perceiver_mmp.training.metrics import evaluate_predictions
from src.hir_perceiver_mmp.training.utils import ensure_dir, get_device, load_checkpoint, set_seed


def train_one_epoch(model, loader, device, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for metric, log_vec, trace_vec, labels, keys in tqdm(loader, desc="[Finetune] Epoch", leave=False):  # noqa: F841
        metric = metric.to(device)
        log_vec = log_vec.to(device)
        trace_vec = trace_vec.to(device)
        labels = labels.to(device).float()

        logits = model.forward_cls(metric, log_vec, trace_vec)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * metric.size(0)
    return total_loss / len(loader.dataset)


def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for metric, log_vec, trace_vec, labels, keys in loader:  # noqa: F841
            metric = metric.to(device)
            log_vec = log_vec.to(device)
            trace_vec = trace_vec.to(device)
            logits = model.forward_cls(metric, log_vec, trace_vec)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy().astype(int)
    return probs, labels_np


def main():
    cfg = config.get_default_config()
    set_seed(cfg.training.seed)

    device = get_device(cfg.training.device)

    data_dir = cfg.data.data_dir
    train_ds, val_ds, test_ds, feature_dims, norm_stats = load_datasets_from_dir(
        data_dir,
        metric_max_t=cfg.data.metric_max_t,
        ratios=cfg.data.train_val_test_ratio,
        split_seed=cfg.data.split_seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.finetune.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.finetune.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.finetune.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
    )

    model = HiRPerceiverMMPModel(
        metric_dim=feature_dims["metric_in"],
        log_dim=feature_dims["log_in"],
        trace_dim=feature_dims["trace_in"],
    ).to(device)

    # 加载预训练 checkpoint
    ckpt_path = cfg.training.pretrain_checkpoint
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    state = ckpt["model"]
    model.load_state_dict(state, strict=False)
    # 重置分类头
    # 使用简单的 xavier 初始化
    for m in model.cls_head.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.finetune.lr)

    ensure_dir(cfg.training.outputs_dir)

    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5

    for epoch in range(1, cfg.finetune.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, device, optimizer)
        print(f"[Finetune] Epoch {epoch} train loss: {avg_loss:.6f}")

        # 验证集评估
        val_probs, val_labels = collect_logits_and_labels(model, val_loader, device)
        val_metrics = evaluate_predictions(val_labels, val_probs, fixed_threshold=None)
        print(
            "[Finetune] Epoch {epoch} VAL | ROC-AUC: {roc:.4f} PR-AUC: {pr:.4f} best_threshold: {thr:.4f} "
            "Precision: {p:.4f} Recall: {r:.4f} F1: {f1:.4f}".format(
                epoch=epoch,
                roc=val_metrics["roc_auc"],
                pr=val_metrics["pr_auc"],
                thr=val_metrics["best_threshold"],
                p=val_metrics["precision"],
                r=val_metrics["recall"],
                f1=val_metrics["f1"],
            )
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_threshold = val_metrics["best_threshold"]

    # 使用在验证集上最优 F1 对应的阈值，在测试集上评估
    test_probs, test_labels = collect_logits_and_labels(model, test_loader, device)
    test_metrics = evaluate_predictions(test_labels, test_probs, fixed_threshold=best_threshold)

    print(
        "[Finetune] TEST (threshold from best VAL epoch={epoch}) | ROC-AUC: {roc:.4f} PR-AUC: {pr:.4f} "
        "threshold: {thr:.4f} Precision: {p:.4f} Recall: {r:.4f} F1: {f1:.4f}".format(
            epoch=best_epoch,
            roc=test_metrics["roc_auc"],
            pr=test_metrics["pr_auc"],
            thr=test_metrics["best_threshold"],
            p=test_metrics["precision"],
            r=test_metrics["recall"],
            f1=test_metrics["f1"],
        )
    )


if __name__ == "__main__":
    main()
