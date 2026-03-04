#!/usr/bin/env bash

# 小工具：将仓库内所有 .sh 脚本的行尾统一为 LF，修复 Windows 上传导致的 CRLF 问题

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "【提示】开始规范化仓库内 .sh 脚本的行尾（CRLF -> LF）..."

# 查找所有 .sh 文件
sh_files=$(find . -type f -name "*.sh")

if [ -z "$sh_files" ]; then
  echo "【提示】未找到任何 .sh 文件，无需处理。"
  exit 0
fi

for f in $sh_files; do
  if [ -f "$f" ]; then
    # 将每行末尾的 CR 去掉，仅保留 LF
    sed -i 's/\r$//' "$f"
    echo "已处理：$f"
  fi
done

echo "【完成】所有 .sh 文件的行尾已统一为 LF。"
echo "如之前运行脚本出现 \$'\\r' 相关错误，请重试脚本执行。"
