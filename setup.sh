#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cuda_variant="${1:-cu126}"
case "$cuda_variant" in
  cu126|cu128) ;;
  *) echo '用法：bash setup.sh [cu126|cu128]；RTX 5090 請用 cu128。' >&2; exit 2 ;;
esac
if ! python3 -m venv .venv; then
  echo "系統缺少 python3-venv，改用專案內的 virtualenv bootstrap。"
  python3 -m pip install --target .bootstrap virtualenv
  PYTHONPATH="$PWD/.bootstrap" python3 -m virtualenv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch==2.7.1+$cuda_variant" "torchvision==0.22.1+$cuda_variant" --index-url "https://download.pytorch.org/whl/$cuda_variant"
.venv/bin/python -m pip install -r requirements.txt
echo '安裝完成。先執行 source .venv/bin/activate，再使用 python run.py。'
