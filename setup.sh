#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! python3 -m venv .venv; then
  echo "系統缺少 python3-venv，改用專案內的 virtualenv bootstrap。"
  python3 -m pip install --target .bootstrap virtualenv
  PYTHONPATH="$PWD/.bootstrap" python3 -m virtualenv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
echo '安裝完成。先執行 source .venv/bin/activate，再使用 python run.py。'
