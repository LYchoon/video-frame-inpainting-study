# 影片抽幀與補幀

一行指令完成抽幀、補幀，結果存入 `outputs/`。

- **LDMVFI：主要模型**，目前支援單幀缺口（`n=1`）。
- **RIFE：比較基準**，速度快，支援連續多幀缺口。
- **VACE：保留選項**。1.3B 在目前人類動作影片的實測效果很差，n=1、n=3 都有明顯失真。可能需要更大的 14B，但效果與顯存需求尚未實測，不能保證改善。

## 一次安裝（Linux）

先取得 private repo 權限，再下載：

```bash
git clone git@github.com:LYchoon/video-frame-inpainting-study.git
cd video-frame-inpainting-study
```

**LDMVFI：2080 已實測。** 需要 Git、uv 或 Python 3.9，以及 CUDA Toolkit 11.8（含 nvcc／NVRTC）：

```bash
bash setup-ldmvfi.sh
source .venv-ldmvfi/bin/activate
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

腳本建立獨立的 Python 3.9／PyTorch 1.11+cu113 環境，不與 RIFE／VACE 混裝。非預設 Toolkit 路徑請在安裝前設定 `CUDA_PATH`。**這套 LDMVFI 舊版環境不支援 5090；5090 移植尚未完成。**

**RIFE／VACE：** Python 3.10–3.12，建議 3.12：

```bash
bash setup.sh cu128          # 5090；2080 改用 cu126
source .venv/bin/activate
```

使用 PyTorch 2.7.1；5090 需支援顯卡與 CUDA 12.8 的驅動。此配置依 [PyTorch 官方 Blackwell 支援](https://pytorch.org/blog/pytorch-2-7/)選定，**5090 尚未實測**。RIFE／VACE 不需另裝 CUDA Toolkit。

## 執行

`m` 是週期長度：**每 m 幀保留前 m−n 幀、移除最後連續 n 幀**，`m > n ≥ 1`。首尾強制保留；尾端不足一個週期也按同一規則處理。省略 `m` 時預設 `n+1`。

```bash
# 主要模型：每 2 幀移除 1 幀（預設 LDMVFI）
python run.py --input video.mp4 --m 2 --n 1

# 每 5 幀移除最後 1 幀
python run.py --input video.mp4 --m 5 --n 1

# 相同抽幀與解析度，使用 RIFE 比較（使用另一個環境）
.venv/bin/python run.py --input video.mp4 --m 2 --n 1 --model rife --max-side 384

# 連續多幀：每 5 幀保留 3 幀、移除 2 幀
.venv/bin/python run.py --input video.mp4 --m 5 --n 2 --model rife

# VACE 14B：待較大顯存環境驗證
.venv/bin/python run.py --input video.mp4 --m 2 --n 1 --model vace --model-size 14b
```

LDMVFI 目前只接受 `n=1`，每個缺口只參考前後保留幀；不以生成幀遞迴補長缺口。連續移除多幀請選 RIFE 或 VACE。舊版 `--mode every --n k` 仍可用，等同每 k 幀移除 1 幀，不能與 `--m` 同時指定。

常用選項：

| 參數 | 預設／用途 |
| --- | --- |
| `--device` | `cuda:0`；指定一張卡，例如 `cuda:2`；LDMVFI 需要 CUDA |
| `--max-side` | LDMVFI：384；其他：512；輸出縮放回原尺寸 |
| `--steps` | LDMVFI：200；VACE：30；RIFE 不使用 |
| `--seed` | 42；LDMVFI 使用 seed + 影格索引，VACE 使用 seed + 視窗序號 |
| `--window-size` | VACE：33，重疊 8 幀；可改 17 降低顯存 |
| `--output-dir` | `outputs`，每次建立獨立子目錄 |

LDMVFI 使用 DDIM、eta=1、EMA、FP32；steps 需整除 1000 且 ≤500，例如 50／100／200。VACE 啟用 CPU offload，2080 使用 FP32；支援原生 BF16 的卡使用 BF16。顯存不足時請明確降低 `--max-side`，程式不會自行改參數。

## 架構與輸出

```text
run.py              # 指令入口
src/                # cli、core、video_io、ldmvfi、rife、vace
outputs/<片名>_<時間>/
  masked.mp4        # 移除位置顯示黑畫面
  restored.mp4      # 補幀結果
  run.json          # m、n、實際移除位置（0 起算）、版本、耗時
```

三個模型都不讀取缺失幀作為推論輸入；只替換缺失幀。保留來源 FPS、影格數、尺寸與音訊時間位置，MP4／AAC 重編碼仍有壓縮差異。輸入需固定 FPS；可變 FPS 會提示先轉檔。權重首次下載後快取；影片、權重及結果不提交 Git。`FRAME_FILL_CACHE` 可指定模型原始碼與權重快取，VACE 權重另使用 `HF_HOME`。

測試：`python -m unittest -v`。實測設定、素材授權與限制見 [VALIDATION.md](VALIDATION.md)。

上游：[LDMVFI](https://github.com/danier97/LDMVFI)、[RIFE](https://github.com/hzwer/ECCV2022-RIFE)、[VACE](https://github.com/Wan-Video/Wan2.1)。使用固定原始碼版本與官方權重，不訓練模型；第三方程式碼留在快取並遵循其授權。
