# 實測紀錄

日期：2026-10-23；主機 `ssh 2080`，RTX 2080 Ti 11 GB，driver 570.124.04。5090、VACE 14B 均尚未實測。

## 人類動作影片

素材：[A Man Dancing in a Room](https://www.pexels.com/video/a-man-dancing-in-a-room-8853811/)，Los Muertos Crew，[Pexels License](https://www.pexels.com/license/)。取第 2–12 秒，轉為 288×512、16 FPS、160 幀、10 秒，無音訊。推論最長邊 384（208×384），seed 42，首尾保留。

| 模型 | 抽幀設定 | 實際移除 | 觀察 |
| --- | --- | --- | --- |
| LDMVFI | m=2,n=1；DDIM 200 steps、eta=1、EMA、FP32 | 79 幀 | 人物與背景明顯比 VACE 1.3B 完整，仍可能有局部模糊 |
| RIFE_m | m=2,n=1 | 79 幀 | 7.398 秒；整體接近原片，快速手腳動作仍有模糊 |
| VACE 1.3B | m=2,n=1；33 幀視窗、30 steps | 79 幀 | 813.542 秒；效果很差，身體、背景有明顯變形與重影 |
| VACE 1.3B | m=4,n=3；其他設定相同 | 119 幀 | 811.356 秒；同樣明顯失真 |

LDMVFI 完整 10 秒影片由獨立實驗腳本完成，中點分別在兩張 2080 Ti 上計算；使用 seed + 影格索引，不合併顯存。約 799 秒的經過時間含快取及分工，**不是單卡速度基準**；PyTorch peak allocated 約 5.00 GiB。RIFE 與 VACE 為單卡；VACE 約 9.09 GiB，RIFE 約 89 MiB。

已建立原片／LDMVFI／RIFE 的相同 n=1 三欄比較；三者均完整解碼為 160 幀、16 FPS、288×512。以上只是單一片段的定性觀察，尚不能推論不同複雜度下的失真臨界值。更大 VACE 可能值得測試，但尚無 14B 效果或可用顯存結論。

## m／n 組合比較

同一段 10 秒影片、最長邊 384；LDMVFI 維持 200 steps。新增組合均由主程式單卡執行；(2,1) 沿用前述基準。

| m | n | 實際移除／160 幀 | 模型 |
| --- | --- | --- | --- |
| 2 | 1 | 79（49.4%） | LDMVFI、RIFE |
| 4 | 1 | 39（24.4%） | LDMVFI、RIFE |
| 8 | 1 | 19（11.9%） | LDMVFI、RIFE |
| 4 | 2 | 79（49.4%） | RIFE |
| 4 | 3 | 119（74.4%） | RIFE |
| 8 | 2 | 39（24.4%） | RIFE |
| 8 | 3 | 59（36.9%） | RIFE |

輸出完整解碼檢查通過。RIFE 在快速手腳動作處，n=2–3 更容易出現模糊、重影。固定 n 時，改 m 主要改變補幀頻率，單一缺口的長度不變；(2,1) 與 (4,2) 移除比例相同，可比較缺口分布的影響。以上仍是單一影片的觀察。

工作站結果：`outputs/mn-study-20261023/results/`，含兩支比較影片、各組 masked／restored／run.json 與 summary.json；影片不提交 Git。

## 整合與參數驗證

- 7 項自動測試在一般環境與 LDMVFI 獨立環境通過：m/n 位置、預設相容、首尾保留、參數錯誤、不讀取缺失幀、視窗平均、分數 FPS、音訊偏移、VFR 拒絕與完整空補幀流程。
- 主程式 LDMVFI：5 幀短片，m=4,n=1，單張 `cuda:2`，200 steps；只移除索引 3，51.418 秒，三個輸出正常。
- 主程式 RIFE：完整 10 秒片段，m=5,n=2，單張 `cuda:0`；移除 63 幀（最後一幀強制保留），6.640 秒。
- 上述兩組輸出均完整解碼驗證影格數、尺寸、FPS 與黑色遮罩位置；缺失幀以外仍從來源直接合成。
- `setup-ldmvfi.sh` 在既有獨立環境重跑成功，`pip check` 通過。一般安裝腳本的 cu126／cu128 參數已檢查；未在 2080 安裝 cu128。

## 固定版本

| 用途 | 環境／模型 |
| --- | --- |
| LDMVFI | Python 3.9、PyTorch 1.11.0+cu113、Lightning 1.7.7、CuPy 11.6.0、CUDA Toolkit 11.8 |
| RIFE／VACE | Python 3.12.3、PyTorch 2.7.1+cu126、Diffusers 0.36.0、Transformers 4.56.2 |
| LDMVFI 原始碼 | `eee2dc3566f21f22787d66b0f0337e48318403c6` |
| LDMVFI checkpoint SHA256 | `7fc97349accb1a79e39e238f6c2cecddb5d481474eb38762848088c80666fd6b` |
| RIFE 原始碼 | `5d8adbdd40e12c2c8f91930eff838aebe561c086` |
| VACE 1.3B revision | `ec4d2cb062b548996b179d493fdd05340de702a1` |

另保留早期 [Sintel](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4) 測試：第 20 秒起 33 幀、854×480、24 FPS、含音訊；[CC BY 3.0](https://durian.blender.org/sharing/)，© Blender Foundation。RIFE／VACE 均保留影格數、FPS、尺寸與音訊時間位置；VACE 另完成跨視窗檢查。動作影片本身無音訊，音訊驗證來自此素材及自動測試。
