# 影片抽幀與補幀

輸入影片、抽幀數 `n` 與模型，一行指令完成抽幀及填補，結果存進 `outputs/`。支援 **Wan2.1 VACE 1.3B／14B** 與 **RIFE_m**；不需要訓練。

## 專案架構

```text
video-frame-inpainting-study/
  src/
    cli.py          # 參數與抽幀／補幀流程
    core.py         # 抽幀規則、遮罩、視窗與影格合成
    video_io.py     # 影片讀取、FPS 檢查與音訊輸出
    vace.py         # VACE 1.3B／14B 推論
    rife.py         # RIFE_m 推論
    __init__.py
  outputs/          # 執行結果；Git 只保留空目錄標記
  run.py            # 一行指令入口
  setup.sh          # 環境安裝（cu126／cu128）
  requirements.txt
  test_run.py
  README.md
  VALIDATION.md
```

請在專案根目錄執行 `python run.py ...`，主要實作集中於 `src/`。

## 一次安裝

建議 Linux、Python **3.12**（支援 3.10–3.12）、NVIDIA GPU。依工作站顯卡選擇安裝版本；程式每次使用指定的一張卡。

| 工作站 | PyTorch / torchvision | CUDA wheel | 安裝指令 |
| --- | --- | --- | --- |
| RTX 5090 | 2.7.1 / 0.22.1 | **cu128（CUDA 12.8）** | `bash setup.sh cu128` |
| `ssh 2080`：RTX 2080 Ti | 2.7.1 / 0.22.1 | cu126（CUDA 12.6） | `bash setup.sh cu126` |

**RTX 5090 安裝指令：**

```bash
git clone git@github.com:LYchoon/video-frame-inpainting-study.git
cd video-frame-inpainting-study
bash setup.sh cu128
source .venv/bin/activate
```

RTX 5090 屬於 Blackwell 架構，請使用 cu128；[PyTorch 2.7 加入 Blackwell 與 CUDA 12.8 支援](https://pytorch.org/blog/pytorch-2-7/)。版本對應依照 [PyTorch 官方安裝指令](https://pytorch.org/get-started/previous-versions/#v271)。腳本中實際安裝的版本為：

```bash
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

若原本已安裝 cu126，在專案目錄執行 `bash setup.sh cu128` 即可切換；腳本會明確指定 `+cu128`，避免已安裝的同版 cu126 被視為滿足需求。RTX 2080 Ti 則使用 `bash setup.sh cu126`；省略參數仍預設 cu126，保留原本行為。

需要先有存取此 private repo 的 GitHub 權限，系統需安裝 Python 及 Git。`setup.sh` 會建立 `.venv`，只安裝選定的 PyTorch 與依賴。RTX 5090 的 Linux 驅動建議使用 **570.124.06 或更新、支援該顯卡的版本**，參考 [NVIDIA CUDA 12.8 Update 1 發行說明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/)；PyTorch wheel 已附 CUDA 執行所需套件，不必另裝系統 CUDA Toolkit。FFmpeg 由 Python 套件提供，不必另外安裝。缺少 `python3-venv` 時，安裝腳本改用專案內的 virtualenv bootstrap，不修改其他專案環境。

5090 安裝完成後，可確認版本並實際執行一次 GPU 運算：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.randn(32,32,device='cuda'); print('GPU OK:', (x@x).mean().item())"
```

預期顯示 `2.7.1+cu128`、`12.8`、RTX 5090，並成功印出 `GPU OK`。**5090 尚未實測**；現有實測紀錄來自 RTX 2080 Ti。

首次執行時會從 Hugging Face／官方 Google Drive 下載模型，RIFE 原始碼從官方 GitHub 下載。請預留數十 GB 磁碟空間與足夠主記憶體；VACE 的 XXL 文字編碼器在 CPU 執行。後續使用本機快取，請保留快取目錄。

## 使用

RTX 5090 建議先用 **VACE 1.3B**、512 最長邊、33 幀視窗、30 steps，與既有測試設定相同；程式會自動使用 BF16。先確認這個指令可跑通：

```bash
python run.py --input video.mp4 --n 3 --model vace --model-size 1.3b --max-side 512 --window-size 33 --steps 30
```

14B 的單卡顯存需求仍需實測，不保證 RTX 5090 的顯存足以執行目前的 CPU offload 配置。

```bash
# 預設：VACE 1.3B；保留 1 幀、連續移除 3 幀
python run.py --input video.mp4 --n 3

# 每 3 幀移除 1 幀
python run.py --input video.mp4 --n 3 --mode every

# 較快的光流補幀基準
python run.py --input video.mp4 --n 3 --model rife

# 使用較大的生成模型
python run.py --input video.mp4 --n 3 --model vace --model-size 14b

# 指定另一張卡
python run.py --input video.mp4 --n 3 --device cuda:1

# 顯存不足時，明確降低推論尺寸及視窗
python run.py --input video.mp4 --n 3 --max-side 256 --window-size 17
```

抽幀位置以下使用 **第 1 幀起算**：

| 模式 | `n=3` 的行為 |
| --- | --- |
| `gap`（預設） | 保留第 1、5、9…幀，移除中間影格 |
| `every` | 移除第 3、6、9…幀 |

**兩種模式都強制保留首尾幀**，即使它們落在原本應移除的位置。`gap` 的 n 必須 ≥1；`every` 的 n 必須 ≥2。`run.json` 記錄實際移除位置，使用 **0 起算**。僅有 2 幀或抽幀規則未選中任何內部影格時，不載入模型，直接輸出。

## 輸出

```text
outputs/<影片名>_<時間>/
  masked.mp4       # 移除位置以黑畫面顯示，保留音訊
  restored.mp4     # 補幀結果，保留音訊
  run.json         # 參數、移除位置、版本、耗時與執行狀態
```

每次建立新目錄，不覆寫舊結果。成功時 `status` 為 `complete`；中途失敗則記錄 `failed` 與錯誤，不應將該目錄視為完成結果。處理過程會暫存原片及生成影格到結果目錄下，結束後自動清除，請預留足夠磁碟空間。

## 參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--input` | 必填 | 固定 FPS 影片路徑 |
| `--n` | 必填 | 依 `--mode` 決定抽幀方式 |
| `--mode` | `gap` | `gap` / `every` |
| `--model` | `vace` | `vace` / `rife` |
| `--model-size` | `1.3b` | VACE 的 `1.3b` / `14b`；RIFE 請省略 |
| `--device` | `cuda:0` | `cuda:N` 或 `cpu`；VACE 在 CPU 非常慢 |
| `--seed` | `42` | VACE 每個視窗使用 seed + 視窗序號；RIFE 不使用隨機取樣 |
| `--max-side` | `512` | 推論最長邊，不放大來源，尺寸向下對齊 16；輸出回到來源尺寸 |
| `--window-size` | `33` | VACE 視窗，必須 ≥9 且為 4k+1，例如 9、17、33、81 |
| `--steps` | `30` | VACE 去噪步數 |
| `--output-dir` | `outputs` | 結果根目錄，以目前工作目錄為基準 |

VACE 固定使用重疊 8 幀的視窗；重疊處只平均生成影格。尾段重複最後一個已遮罩的輸入影格補齊，輸出裁回原長度。保留幀在影片編碼前原樣取自來源；MP4 重新編碼仍會有壓縮差異。VACE 在支援原生 BF16 的卡使用 BF16，2080 Ti 使用 FP32 避免 FP16 溢位；VAE、文字編碼器使用 FP32，GPU 推論啟用 CPU offload。

RIFE 使用官方 RIFE_m、任意時間點推論，每個缺口只參考兩端保留幀；不使用遞迴生成幀作為端點。RIFE 的 `--steps`、`--window-size`、`--seed` 不影響結果。兩個模型均不讀取被移除的原始影格作為模型輸入；原始影片只用於保留幀與音訊。

VACE 使用固定、中性的動作提示詞，完整記錄於 `run.json`，不從原片的隱藏影格產生描述。長缺口、跨鏡頭切換與跨視窗動作可能不連續，這正是可以觀察的補幀失真；程式不會判定「自然程度」。

來源必須是固定 FPS。若偵測到可變 FPS，請先轉檔，例如：

```bash
ffmpeg -i input.mp4 -vf fps=30 -c:v libx264 -c:a aac input_cfr.mp4
```

一般影片保留 FPS、影格數、畫面尺寸及音訊時間位置；音訊重新編碼為 AAC，超出影片長度的音訊裁切。第一版不保留字幕、章節、HDR 或旋轉顯示中繼資料，請先轉成一般 SDR、實際像素方向正確的影片。

## 測試

```bash
python -m unittest -v
```

自動測試不下載模型，涵蓋抽幀位置、首尾保留、遮罩輸入無答案洩漏、視窗覆蓋與平均、未移除幀保留、錯誤參數、分數 FPS、音訊偏移、VFR 拒絕及無須補幀的完整指令流程。

模型實測紀錄見 [VALIDATION.md](VALIDATION.md)。**VACE 14B 尚未實測**；已提供官方 checkpoint 切換，需要資源足夠的機器驗證。

## 模型與素材來源

- [Wan2.1 VACE](https://github.com/Wan-Video/Wan2.1)、[Diffusers VACE](https://huggingface.co/docs/diffusers/api/pipelines/wan#wanvacepipeline)：1.3B／14B 官方 Diffusers checkpoint；記錄每次使用的模型 revision。
- [RIFE](https://github.com/hzwer/ECCV2022-RIFE)：使用固定原始碼 commit `5d8adbdd40e12c2c8f91930eff838aebe561c086` 與 README 連結的官方 RIFE_m 權重；記錄 checkpoint SHA256。第三方原始碼留在快取，遵循上游 MIT 授權。
- 實測素材：[Sintel 預告片](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4)，[CC BY 3.0 授權說明](https://durian.blender.org/sharing/)；© copyright Blender Foundation | durian.blender.org。影片及權重不提交至本 repo。

Hugging Face 權重快取可用 `HF_HOME` 指定；RIFE 與固定提示詞的編碼快取可用 `FRAME_FILL_CACHE` 指定。提示詞編碼依模型 revision 與文字內容快取，後續執行可省去 XXL 編碼器載入。請勿將影片、權重、快取或輸出提交至 Git。
