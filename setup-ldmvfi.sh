#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -ne 0 ]; then
  echo '用法：bash setup-ldmvfi.sh；適用已測試的 2080 環境，不支援 5090。' >&2
  exit 2
fi
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-11.8}"
if [ ! -x "$CUDA_PATH/bin/nvcc" ]; then
  echo 'LDMVFI 需要 CUDA Toolkit 11.8（含 nvcc / NVRTC）；請先設定 CUDA_PATH。' >&2
  exit 2
fi
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.9 --seed --allow-existing .venv-ldmvfi
elif command -v python3.9 >/dev/null 2>&1; then
  python3.9 -m venv .venv-ldmvfi
else
  echo '需要 uv 或 Python 3.9；安裝後重新執行，參閱 README。' >&2
  exit 2
fi
# New pip rejects Lightning 1.7.7 metadata; new setuptools removes pkg_resources.
.venv-ldmvfi/bin/python -m pip install pip==23.3.2 setuptools==65.5.1 wheel
.venv-ldmvfi/bin/python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --index-url https://download.pytorch.org/whl/cu113
.venv-ldmvfi/bin/python -m pip install -r requirements-ldmvfi.txt
# Upstream's non-editable wheel omits the taming package; keep the pinned source.
.venv-ldmvfi/bin/python -m pip install --no-deps -e 'git+https://github.com/CompVis/taming-transformers.git@3ba01b241669f5ade541ce990f7650a3b8f65318#egg=taming-transformers'
echo '安裝完成。source .venv-ldmvfi/bin/activate；執行時仍需設定 CUDA_PATH 與 LD_LIBRARY_PATH，參閱 README。'
