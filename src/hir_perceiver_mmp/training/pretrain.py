import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.hir_perceiver_mmp.data.dataset import collate_fn, load_datasets_from_dir
from src.hir_perceiver_mmp.models.model import HiRPerceiverMMPModel
from src.hir_perceiver_mmp.training import metrics as metrics_mod
from src.hir_perceiver_mmp.training.utils import (
    create_two_masked_views,
    ensure_dir,
    get_device,
    save_checkpoint,
    set_seed,
)


def train_epoch(model, loader, device, optimizer, cfg):
    model.train()
    mse_loss = torch.nn.MSELoss()
    total_loss = 0.0
    total_metric = 0.0
    total_log = 0.0
    total_trace = 0.0
    n_samples = 0
    for batch in tqdm(loader, desc="[Pretrain] Epoch", leave=False):
        metric, log_vec, trace_vec, labels, keys = batch  # noqa: F841
        metric = metric.to(device)
        log_vec = log_vec.to(device)
        trace_vec = trace_vec.to(device)

        (m1, l1, t1), (m2, l2, t2) = create_two_masked_views(metric, log_vec, trace_vec, cfg)

        metric_rec1, log_rec1, trace_rec1, z1 = model.forward_pretrain(m1, l1, t1)
        metric_rec2, log_rec2, trace_rec2, z2 = model.forward_pretrain(m2, l2, t2)

        metric_flat = metric.view(metric.size(0), -1)
        log_flat = log_vec
        trace_flat = trace_vec

        metric_loss1 = mse_loss(metric_rec1, metric_flat)
        log_loss1 = mse_loss(log_rec1, log_flat)
        trace_loss1 = mse_loss(trace_rec1, trace_flat)
        L_rec1 = cfg.pretrain.lambda_metric * metric_loss1 + cfg.pretrain.lambda_log * log_loss1 + cfg.pretrain.lambda_trace * trace_loss1

        metric_loss2 = mse_loss(metric_rec2, metric_flat)
        log_loss2 = mse_loss(log_rec2, log_flat)
        trace_loss2 = mse_loss(trace_rec2, trace_flat)
        L_rec2 = cfg.pretrain.lambda_metric * metric_loss2 + cfg.pretrain.lambda_log * log_loss2 + cfg.pretrain.lambda_trace * trace_loss2

        # 按照两视图平均后的三项重建损失（已乘 lambda）做统计，方便诊断哪一块贡献最大
        batch_size = metric.size(0)
        metric_rec_batch = 0.5 * cfg.pretrain.lambda_metric * (metric_loss1 + metric_loss2)
        log_rec_batch = 0.5 * cfg.pretrain.lambda_log * (log_loss1 + log_loss2)
        trace_rec_batch = 0.5 * cfg.pretrain.lambda_trace * (trace_loss1 + trace_loss2)

        rec_loss = metric_rec_batch + log_rec_batch + trace_rec_batch

        cos_sim = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean()
        cons_loss = 1.0 - cos_sim

        loss = rec_loss + cfg.pretrain.lambda_consistency * cons_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_metric += metric_rec_batch.item() * batch_size
        total_log += log_rec_batch.item() * batch_size
        total_trace += trace_rec_batch.item() * batch_size
        n_samples += batch_size

    avg_total = total_loss / n_samples
    avg_metric = total_metric / n_samples
    avg_log = total_log / n_samples
    avg_trace = total_trace / n_samples
    return avg_total, avg_metric, avg_log, avg_trace


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
        batch_size=cfg.pretrain.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
    )

    model = HiRPerceiverMMPModel(
        metric_dim=feature_dims["metric_in"],
        log_dim=feature_dims["log_in"],
        trace_dim=feature_dims["trace_in"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain.lr)

    ensure_dir(cfg.training.outputs_dir)

    for epoch in range(1, cfg.pretrain.epochs + 1):
        avg_loss, avg_metric, avg_log, avg_trace = train_epoch(model, train_loader, device, optimizer, cfg)
        print(
            "[Pretrain] Epoch {epoch} | total: {tot:.6f} metric_rec: {m:.6f} log_rec: {l:.6f} trace_rec: {t:.6f}".format(
                epoch=epoch,
                tot=avg_loss,
                m=avg_metric,
                l=avg_log,
                t=avg_trace,
            )
        )

    ckpt_path = cfg.training.pretrain_checkpoint
    save_checkpoint(ckpt_path, model.state_dict(), extra={"feature_dims": feature_dims})
    print(f"预训练完成，checkpoint 保存在: {ckpt_path}")


if __name__ == "__main__":
    main()
