#!/usr/bin/env bash

# 一键示例运行脚本：生成样例数据 -> 预训练 -> 微调，并将关键日志写入 outputs/smoke_run.log

set -euo

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs"
DATA_DIR="data/sample_run"
LOG_FILE="${LOG_DIR}/smoke_run.log"

mkdir -p "$LOG_DIR" "$DATA_DIR"
: > "$LOG_FILE"

# 选择 Python 解释器：优先 python3，其次 python
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  MSG="[错误] 未检测到 Python，请安装 python3 或提供 python 命令"
  if command -v tee >/dev/null 2>&1; then
    echo "$MSG" | tee -a "$LOG_FILE"
  else
    echo "$MSG" >> "$LOG_FILE"
  fi
  exit 1
fi

# 稳健日志输出：尽量同时输出到终端和日志文件
log_echo() {
  local msg="$1"
  if command -v tee >/dev/null 2>&1; then
    echo "$msg" | tee -a "$LOG_FILE"
  else
    echo "$msg" >> "$LOG_FILE"
  fi
}

log_run() {
  if command -v tee >/dev/null 2>&1; then
    "$@" 2>&1 | tee -a "$LOG_FILE"
  else
    "$@" >>"$LOG_FILE" 2>&1
  fi
}

log_echo "[Quickstart] 仓库根目录：${ROOT_DIR}"

log_echo "[Quickstart] 生成示例数据..."
log_run "$PYTHON" -m src.hir_perceiver_mmp.data.sample_data_generator --output_dir "$DATA_DIR"

log_echo "[Quickstart] 预训练（C3-MMP）..."
log_run "$PYTHON" -m src.hir_perceiver_mmp.training.pretrain

log_echo "[Quickstart] 微调并评估..."
log_run "$PYTHON" -m src.hir_perceiver_mmp.training.finetune
