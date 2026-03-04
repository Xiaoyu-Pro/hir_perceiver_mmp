# 多模态异常检测示例工程（HiR-Perceiver + System Causal Reasoner + C3-MMP）

本工程实现了一个**可在 CPU 上运行的最小多模态异常检测示例**，包含：

- Metric/Log/Trace 三模态对齐数据读取与归一化
- HiR-Perceiver 多模态融合骨干
- System Causal Reasoner 系统化因果推理模块
- C3-MMP 预训练机制（结构化 Mask + Two-view 一致性）
- 预训练 → 微调 的完整训练与评估流程

生成的示例数据规模很小（几十个窗口），可在普通 CPU 环境数分钟内完成一轮预训练 + 微调。

---

## 1. 环境依赖

- Python >= 3.9
- 主要第三方库：
  - `torch`
  - `numpy`
  - `scikit-learn`
  - `tqdm`
  - `pyyaml`（当前示例未强依赖，可按需安装）

在仓库根目录（`hir_perceiver_mmp/`）下执行：

```bash
pip install -r requirements.txt
```

> 注：本示例默认使用 CPU 运行，如需使用 GPU，可在 `config.py` 中将 `TrainingConfig.device` 修改为 `"cuda"`，并确保本机安装了对应的 CUDA 版 PyTorch。

---

## 2. 目录结构

```text
hir_perceiver_mmp/
  ├── config.py                # 集中配置：数据、backbone、reasoner、预训练、微调
  ├── requirements.txt
  ├── README_zh.md             # 本文件
  ├── scripts/
  │   ├── run_quickstart.sh        # 一键示例：生成数据 + 预训练 + 微调
  │   └── normalize_sh_endings.sh # 修复 .sh 脚本行尾（Windows CRLF -> Linux LF）
  ├── src/
  │   └── hir_perceiver_mmp/
  │       ├── __init__.py
  │       ├── data/
  │       │   ├── __init__.py
  │       │   ├── dataset.py               # JSON 读取、对齐、Dataset & collate_fn
  │       │   └── sample_data_generator.py # 生成 metric/log/trace/label 示例 JSON
  │       ├── models/
  │       │   ├── __init__.py
  │       │   ├── tokenizers.py            # Metric/Log/Trace tokenizer
  │       │   ├── fusion.py                # HiR-Perceiver 融合骨干
  │       │   ├── reasoner.py              # System Causal Reasoner
  │       │   ├── heads.py                 # 分类头与预训练重建头
  │       │   └── model.py                 # 封装 forward_backbone/cls/pretrain
  │       └── training/
  │           ├── __init__.py
  │           ├── utils.py                 # 随机种子、设备、mask 生成与 checkpoint
  │           ├── metrics.py               # ROC-AUC/PR-AUC/F1 等计算
  │           ├── pretrain.py              # C3-MMP 预训练
  │           └── finetune.py              # 从预训练加载并进行分类微调
  └── outputs/                  # 运行后生成：checkpoint、smoke_run.log 等
```

---

## 3. 数据格式与示例生成

### 3.1 JSON 数据约定（窗口级对齐）

所有文件以 `window_id` 作为主键（字符串："0","1",...），并取四份数据的交集对齐：

- `metric.json`：
  - `{"window_id": {"time_range": "...", "data": [{"timestamp": ..., "kpiA": ..., ...}, ...]}}`
  - 每个窗口是长度可变时序，工程中会根据第一个样本推断 feature keys，并在 `Dataset` 内部 pad/truncate 到固定 `metric_max_t`，得到 `[Tmax, F]`。
- `log.json`：
  - `metadata.sorted_eventids`、`metadata.sorted_levels`
  - `data[window_id] = {"eventid_count": [...], "levels": [...], "total_count": ...}`
  - 拼接为 `log_vec = [total_count] + eventid_count + levels`，得到 `[D_log]`。
- `trace.json`：
  - `metadata.sorted_names`
  - `data[window_id] = {"counts": [...], "durations": [...], "total_count": ..., "total_duration": ...}`
  - 拼接为 `trace_vec = [total_count, total_duration] + counts + durations`，得到 `[D_trace]`。
- `label.json`：
  - `data[window_id] = {"time_range": "...", "label": bool}`
  - 训练时转换为 `{0,1}`。

归一化策略（可在 `config.py` 中开关）：

- metric：按 train split 统计 `mean/std` 做 z-score，应用于 `[Tmax,F]`。
- log / trace：对原始向量做 `log1p(max(x,0))` 抑制长尾。

### 3.2 生成极小示例数据

仓库中提供 `sample_data_generator.py`，会随机生成一组**对齐完好的** JSON：

- 每 5 个窗口构造一个异常（指标抬升、日志 ERROR 激增、调用时长增加）。

手动生成示例数据（可选）：

```bash
cd hir_perceiver_mmp
python3 -m src.hir_perceiver_mmp.data.sample_data_generator --output_dir data/sample_run
```

> 该命令会在 `data/sample_run` 下生成 `metric.json`、`log.json`、`trace.json`、`label.json` 四个文件。

---

## 4. 模型与训练流程概览

### 4.1 Tokenizers

- **MetricTokenizer**：
  - 输入：`metric: [B, Tmax, F]`
  - 线性投影 `Linear(F -> d_model)` 得到 `metric_tokens: [B, Tmax, d_model]`。
- **LogTokenizer / TraceTokenizer**（Vector-as-Tokens）：
  - 输入向量 `[B, D]`，为每个维度学习一个 `Emb[i] ∈ R^{d_model}`，输出
  - `tokens[:, i, :] = Emb[i] * value_i`，实现 value-gated embedding，形状 `[B, D, d_model]`。

### 4.2 HiR-Perceiver Fusion（backbone）

- 为三模态分别维护一组 local latents `Z_m_local/Z_l_local/Z_t_local`，通过 cross-attention 从对应 tokens 中读取信息。
- 维护一组 global latents `Z_global`，依次从三种 local latents 中读取，并堆叠若干层 self-attention。
- pooling 得到 `z_global = mean(Z_global, dim=latent_len)`。
- 同时根据输入构建模态质量特征：
  - metric：零值比例 & 方差
  - log：稀疏率 & top1 占比
  - trace：稀疏率 & 总时长
- 通过轻量 MLP + Sigmoid 得到模态 gate `g_m/g_l/g_t`，对 global 更新做缩放；trace 另有 `b_t` 作为 bias，最终使用 `g_t * b_t` 注入 trace 证据信号。

### 4.3 System Causal Reasoner

- 对 `z_global` 做三路投影：`z_state/z_behavior/z_path`，分别对应系统状态、异常证据、调用路径。
- 循环 K 层：
  - `z_state ← Attn(z_state, z_behavior)`
  - `z_behavior ← Attn(z_behavior, z_path)`
  - `z_path ← Attn(z_path, z_state)`
- 将三者拼接后输入 MLP，得到推理后的 `z_reasoned`。
- 在当前示例中，预训练使用 `z_global`；微调阶段默认启用 Reasoner 作为 post-fusion 插件。

### 4.4 Heads

- **分类头**：MLP + Sigmoid，输出窗口级异常概率。
- **预训练重建头**：从 `z_global` 重建
  - `metric_flat: [B, Tmax*F]`
  - `log_vec: [B, D_log]`
  - `trace_vec: [B, D_trace]`

### 4.5 C3-MMP 预训练

- 结构化 Mask：
  - Metric block：沿时间轴随机连续片段置零。
  - Log burst：对向量中 top-k 大的维度置零。
  - Trace edge-drop：除前缀统计外随机置零 span 统计维度。
- Two-view：对同一 batch 生成两份 masked 视角 `(view1, view2)`，同一模型分别前向：
  - 重建损失：
    - `L_metric = MSE(metric_rec, metric_flat)`
    - `L_log = MSE(log_rec, log_vec)`
    - `L_trace = MSE(trace_rec, trace_vec)`
    - `L_rec = λm L_metric + λl L_log + λt L_trace`
  - 一致性损失：`L_cons = 1 - cosine(z(view1), z(view2))`
  - 总损失：`L_pre = 0.5*(L_rec(view1)+L_rec(view2)) + λc*L_cons`

### 4.6 微调与评估

- Stage-1：`pretrain.py` 训练 Tokenizers + Fusion（+Reasoner）+ 重建头，并保存 checkpoint。
- Stage-2：`finetune.py` 加载 checkpoint，重置分类头，在同一数据集上进行窗口级异常检测训练。
- 每个 epoch 在验证集上输出：
  - ROC-AUC
  - PR-AUC
  - best threshold（扫描 F1 最大）
  - 对应 Precision / Recall / F1
- 在测试集上使用 **验证集上最优 F1 对应的阈值** 固定评估，避免信息泄露。

---

## 5. 一键最小示例运行

在仓库根目录执行：

```bash
cd hir_perceiver_mmp
bash scripts/run_quickstart.sh
```

脚本将依次执行：

1. 生成示例数据：
   - 调用 `python3 -m src.hir_perceiver_mmp.data.sample_data_generator --output_dir data/sample_run`
2. 预训练（C3-MMP，1 epoch）：
   - 调用 `python3 -m src.hir_perceiver_mmp.training.pretrain`
3. 微调（1 epoch）并评估：
   - 调用 `python3 -m src.hir_perceiver_mmp.training.finetune`

所有标准输出会同时写入 `outputs/smoke_run.log`，其中包含：

- 预训练阶段每个 epoch 的平均损失
- 微调阶段每个 epoch 在验证集上的：ROC-AUC / PR-AUC / best threshold / Precision / Recall / F1
- 最终在测试集上的固定阈值评估结果

---

## 6. 配置与扩展建议

- 所有核心可调参数位于 `config.py`：
  - `DataConfig`：数据目录、`metric_max_t`、归一化开关、划分比例与随机种子。
  - `BackboneConfig`：latent 长度、层数、注意力头数、dropout、reliability gate / trace bias 开关。
  - `ReasonerConfig`：是否启用、深度 K、隐藏维度与注意力头数。
  - `PretrainConfig`：学习率、batch size、epoch 数、各模态重建权重、mask 比例、一致性损失权重。
  - `FinetuneConfig`：学习率、batch size、epoch 数。
  - `TrainingConfig`：设备（CPU/GPU）、随机种子、DataLoader 并行数、输出目录与 checkpoint 路径。

你可以在保持接口不变的前提下：

- 替换为真实业务数据（只要遵守 JSON 格式与字段约定）。
- 调整 `d_model`、latent 数量和 Reasoner 深度，做结构消融与性能对比。
- 增强 C3-MMP 中的 mask 策略（例如按业务规则设计缺失模式）。

---

## 7. 关键命令速查

在仓库根目录：

```bash
# 安装依赖
pip install -r requirements.txt

# 仅生成示例数据
# (脚本会优先使用 python3；若无，则回退到 python)
python3 -m src.hir_perceiver_mmp.data.sample_data_generator --output_dir data/sample_run

# 仅运行预训练
python3 -m src.hir_perceiver_mmp.training.pretrain

# 在已有预训练 checkpoint 上运行微调 + 评估
python3 -m src.hir_perceiver_mmp.training.finetune

# 一键完整冒烟（数据生成 + 预训练 + 微调 + 评估）
bash scripts/run_quickstart.sh
```

---

## 8. 常见问题

### Q1: 运行 `bash scripts/run_quickstart.sh` 时报错 `scripts/run_quickstart.sh: line X: $'\r': command not found`

**原因**：
这是典型的 Windows/Linux 行尾符（End of Line, EOL）问题。Windows 使用 `CRLF`（回车+换行）作为行尾，而 Linux/macOS 使用 `LF`（换行）。当你在 Windows 环境下克隆或编辑了 `.sh` 脚本，再到 Linux 环境中执行时，Shell 会将每一行末尾多余的 `CR`（Carriage Return, `\r`）字符误认为是命令的一部分，从而导致解析错误。

**解决方案**：
我们提供了一个辅助脚本来批量修复此问题。在仓库根目录运行：

```bash
bash scripts/normalize_sh_endings.sh
```
该脚本会自动查找并转换所有 `.sh` 文件的行尾为 `LF`。完成后，请重新运行 `bash scripts/run_quickstart.sh`。

### Q2: 关于 `python` 与 `python3` 命令的选择

**说明**：
为保证最大兼容性，`scripts/run_quickstart.sh` 脚本内部已实现稳健的 Python 解释器选择逻辑：
1. **优先使用 `python3`**：如果系统中 `python3` 命令可用，则所有 Python 相关操作都将通过 `python3` 执行。
2. **自动回退 `python`**：如果 `python3` 不存在，脚本会尝试使用 `python` 命令。
3. **失败则提示**：如果两者都未找到，脚本会输出中文错误信息并退出。

因此，你无需担心本地环境的 Python 命令是 `python` 还是 `python3`，直接运行 `bash scripts/run_quickstart.sh` 即可。本 README 中的命令示例统一使用 `python3`，以遵循最佳实践。
