# 實測紀錄

日期：2026-10-23。主機：`ssh 2080`；RTX 2080 Ti（每張 11 GB）、driver 570.124.04、Python 3.12.3。每次推論使用單張卡，未將 8 張卡的顯存合併。

素材為 [Sintel 預告片](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4) 20 秒起的 1.375 秒片段，854×480、24 FPS、33 幀、含音訊。[CC BY 3.0](https://durian.blender.org/sharing/)；© copyright Blender Foundation | durian.blender.org。測試影片與模型不提交 Git。

## 自動檢查

`python -m unittest -v`：6 項測試通過。涵蓋兩種抽幀規則、首尾保留、錯誤參數、未移除影格在合成前原樣保留、隱藏影格不進入模型條件、重疊平均與尾段補齊、30000/1001 FPS、音訊偏移保留、VFR 拒絕及無須補幀的完整輸出流程。

## 模型測試

測試設定：`n=3`、`gap`、`max-side=512`；33 幀中填補 24 幀。VACE 使用 33 幀視窗、30 steps、seed 42。

| 模型 | 結果 | 耗時／記憶體 |
| --- | --- | --- |
| RIFE_m | 已通過，輸出完整影格與音訊；另驗證 `cuda:1` | 快取後 4.533 秒，首次含原始碼／權重下載 36.544 秒；PyTorch peak allocated 約 127 MiB |
| VACE 1.3B | 已通過，預設指令；FP32、CPU offload | 256.577 秒，去噪部分約 180 秒；PyTorch peak allocated 約 9.50 GiB |
| VACE 14B | 未實測 | 僅確認官方 checkpoint 路徑，需較大顯存環境測試 |

另已實測 RIFE `every` 模式與 `--device cpu`：移除 10 幀，完成耗時 8.682 秒。RIFE 為光流補幀基準，不是 diffusion 模型。

兩個模型的 `masked.mp4` 與 `restored.mp4` 均已完整解碼：33 幀、24 FPS、854×480、音訊長度 1.375 秒，時間位置與來源一致。缺失位置為黑畫面，補回位置有畫面內容，並已檢視抽樣畫格。

另實際執行 VACE 快取與跨視窗檢查：18 幀短片、17 幀視窗、重疊 8 幀、最長邊 128、1 step、`cuda:2`，正常產生兩個視窗（第 1–17、10–18 幀）並輸出原始 18 幀。此檢查只驗證流程，不評估 1 step 的生成品質。

最終依賴：PyTorch 2.7.1+cu126、Diffusers 0.36.0、Transformers 4.56.2、PyAV 15.1.0；`pip check` 通過。VACE 模型 revision：`ec4d2cb062b548996b179d493fdd05340de702a1`。

RTX 5090 的安裝配置另提供 PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128；其他依賴維持相同版本。此配置依官方 Blackwell 支援選定，尚未在 RTX 5090 執行模型驗證，以上耗時與記憶體數字不適用於 5090。

整理為 `src/` 與 `outputs/` 後，重新通過 6 項自動測試、VACE 模組匯入及 RIFE 真實補幀；根目錄 `python run.py ...` 指令維持可用。安裝腳本以替身程式驗證預設 cu126、明確 cu126、明確 cu128 與錯誤參數拒絕，未在 2080 主機實際安裝 cu128。

耗時包含程式內解碼、模型載入與編碼，首次下載、磁碟快取與 CPU offload 會影響速度；PyTorch peak allocated 不等於整張顯卡的總占用。本驗收確認程式能執行與輸出，不代表補幀自然度的研究結論。
