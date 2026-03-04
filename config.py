from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    # 数据来源模式："sample" 使用 data_dir 下的示例数据；"real" 使用真实数据路径组件拼接
    mode: str = "sample"  # "sample" 或 "real"

    # 示例数据目录（相对于仓库根目录），用于 quickstart/demo
    data_dir: str = "data/sample_run"

    # 真实数据根目录与子路径组件（在 mode="real" 时生效）
    real_root_dir: str = "/home/admin/projects/MultiFusion/code"
    dataset_name: str = "dataA"
    time_range: str = "2days"
    window_type: str = "10s_5s"
    service_name: str = "mobservice1"

    # 真实数据文件名（在 mode="real" 时生效）
    metric_filename: str = "processed_metric.json"
    log_filename: str = "processed_log.json"
    trace_filename: str = "processed_trace.json"
    label_filename: str = "processed_label.json"

    # 通用数据相关配置
    metric_max_t: int = 16
    use_metric_zscore: bool = True
    use_log_log1p: bool = True
    use_trace_log1p: bool = True
    train_val_test_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    split_seed: int = 42


@dataclass
class BackboneConfig:
    d_model: int = 64
    n_heads: int = 4
    local_latent_len: int = 8
    global_latent_len: int = 8
    global_self_layers: int = 1
    dropout: float = 0.1
    use_reliability_gate: bool = True
    use_trace_bias: bool = True


@dataclass
class ReasonerConfig:
    enabled: bool = True
    depth: int = 2
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1


@dataclass
class PretrainConfig:
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-3
    lambda_metric: float = 1.0
    lambda_log: float = 1.0
    lambda_trace: float = 1.0
    lambda_consistency: float = 0.1
    metric_block_mask_ratio: float = 0.3
    log_burst_mask_ratio: float = 0.2
    trace_edge_drop_ratio: float = 0.2


@dataclass
class FinetuneConfig:
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-3


@dataclass
class TrainingConfig:
    device: str = "cpu"  # "cpu" 或 "cuda"
    seed: int = 1234
    num_workers: int = 0
    outputs_dir: str = "outputs"
    pretrain_checkpoint: str = "outputs/pretrain_checkpoint.pth"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> Config:
    """返回默认配置。"""
    return Config()
