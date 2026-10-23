"""Frame selection and merging; no model dependencies."""
import numpy as np
from PIL import Image


def removal_mask(count, n, mode):
    if n < 1 or (mode == "every" and n < 2):
        raise ValueError("gap 的 n 必須 >= 1；every 的 n 必須 >= 2。")
    index = np.arange(count)
    if mode == "gap":
        mask = index % (n + 1) != 0
    elif mode == "every":
        mask = (index + 1) % n == 0  # human-facing positions start at 1
    else:
        raise ValueError("未知抽幀模式。")
    if count:
        mask[0] = mask[-1] = False
    return mask


def windows(count, size, overlap=8):
    if size < 9 or (size - 1) % 4 or overlap >= size:
        raise ValueError("window-size 必須 >= 9 且為 4k+1，例如 9、17、33。")
    start = 0
    while start < count:
        end = min(start + size, count)
        yield start, end
        if end == count:
            break
        start += size - overlap


def inference_size(width, height, max_side, multiple=16):
    scale = min(1, max_side / max(width, height))
    return tuple(max(multiple, int(v * scale) // multiple * multiple)
                 for v in (width, height))


def condition_window(frames, mask, start, end, size, dimensions):
    """Never read removed pixels, including for temporal tail padding."""
    video, masks = [], []
    for i in range(start, end):
        missing = bool(mask[i])
        image = (Image.new("RGB", dimensions, (128, 128, 128)) if missing
                 else Image.fromarray(frames[i]).resize(dimensions, Image.Resampling.LANCZOS))
        video.append(image)
        masks.append(Image.new("L", dimensions, 255 if missing else 0))
    while len(video) < size:
        video.append(video[-1].copy())
        masks.append(masks[-1].copy())
    return video, masks


def add_prediction(sums, counts, index, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError("模型產生 NaN/Inf，已停止；請檢查模型精度與硬體設定。")
    sums[index] += np.clip(prediction, 0, 1)
    counts[index] += 1


def output_frame(original, missing, prediction=None):
    if not missing:
        return original  # exact preservation before video encoding
    return np.asarray(Image.fromarray(prediction).resize(
        (original.shape[1], original.shape[0]), Image.Resampling.LANCZOS))
