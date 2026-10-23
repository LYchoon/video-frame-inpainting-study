"""One command: remove frames, reconstruct them, and save outputs."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from .core import inference_size, output_frame, removal_mask, windows
from .video_io import decode_video, encode_video


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="影片抽幀與補幀：LDMVFI / RIFE / VACE")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, help="每 m 幀移除最後連續 n 幀；m > n，預設 n+1")
    parser.add_argument("--mode", choices=["gap", "every"], default="gap")
    parser.add_argument("--model", choices=["ldmvfi", "rife", "vace"], default="ldmvfi")
    parser.add_argument("--model-size", choices=["1.3b", "14b"], default="1.3b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--window-size", type=int, default=33)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if args.max_side is None:
        args.max_side = 384 if args.model == "ldmvfi" else 512
    if args.steps is None:
        args.steps = 200 if args.model == "ldmvfi" else 30
    if not args.input.is_file():
        parser.error("找不到輸入影片。")
    try:
        removal_mask(2, args.n, args.mode, args.m)
        if args.mode == "gap" and args.m is None:
            args.m = args.n + 1
        list(windows(2, args.window_size))
        if args.max_side < 32 or args.steps < 1 or args.seed < 0:
            raise ValueError("max-side 必須 >= 32；steps 必須 >= 1；seed 必須 >= 0。")
        if args.model != "vace" and args.model_size != "1.3b":
            raise ValueError("model-size 僅用於 VACE；其他模型請省略此參數。")
        if args.model == "ldmvfi":
            if args.mode == "gap" and args.n != 1:
                raise ValueError("LDMVFI 目前支援單幀缺口：gap 請用 --n 1，或使用 every 模式；連續多幀請選 RIFE / VACE。")
            if args.steps > 500 or 1000 % args.steps:
                raise ValueError("LDMVFI steps 必須整除 1000 且 <= 500，例如 50、100、200。")
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device 請使用 cpu 或 cuda:N。")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("找不到 CUDA GPU；請安裝 CUDA 版 PyTorch，或指定 --device cpu。")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"不存在的 GPU：{device}。")
        args.device = str(device)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.input.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = args.output_dir / name
    output.mkdir(exist_ok=False)
    packages = ["torch", "torchvision", "numpy", "av", "Pillow", "imageio-ffmpeg"]
    if args.model == "ldmvfi":
        packages += ["pytorch-lightning", "cupy-cuda11x", "timm", "omegaconf"]
    elif args.model == "vace":
        packages += ["diffusers", "transformers", "accelerate"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass  # A no-op run needs no model-specific dependencies.
    record = {"status": "running", "parameters": {k: str(v) if isinstance(v, Path) else v
               for k, v in vars(args).items()}, "versions": versions}
    try:
        # Disk-backed frames avoid loading the whole source into RAM.
        with tempfile.TemporaryDirectory(prefix="frames-", dir=output) as scratch, ExitStack() as cleanup:
            frames, info = decode_video(args.input, scratch)
            cleanup.callback(frames._mmap.close)
            mask = removal_mask(len(frames), args.n, args.mode, args.m)
            dimensions = inference_size(info["width"], info["height"], args.max_side)
            record.update(source=info, removed_frame_indices_0based=np.flatnonzero(mask).tolist(),
                          inference_dimensions=list(dimensions))
            print(f"影片 {len(frames)} 幀；移除 {mask.sum()} 幀；推論尺寸 {dimensions}", flush=True)
            sums = np.memmap(Path(scratch) / "generated.f32", mode="w+", dtype=np.float32,
                             shape=(len(frames), dimensions[1], dimensions[0], 3))
            cleanup.callback(sums._mmap.close)
            sums[:] = 0
            counts = np.zeros(len(frames), dtype=np.uint32)
            if mask.any():
                if args.model == "vace":
                    from .vace import reconstruct
                elif args.model == "ldmvfi":
                    from .ldmvfi import reconstruct
                else:
                    from .rife import reconstruct
                record["model"] = reconstruct(frames, mask, sums, counts, dimensions, args)
            if np.any(mask & (counts == 0)):
                raise RuntimeError("部分缺失幀未被模型填補，已停止輸出。")
            black = np.zeros_like(frames[0])
            encode_video(output / "masked.mp4", (black if mask[i] else frames[i]
                         for i in range(len(frames))), args.input.resolve(), info)

            def restored_frames():
                for i in range(len(frames)):
                    prediction = None
                    if mask[i]:
                        prediction = np.rint(np.clip(sums[i] / counts[i], 0, 1) * 255).astype(np.uint8)
                    yield output_frame(frames[i], mask[i], prediction)

            encode_video(output / "restored.mp4", restored_frames(), args.input.resolve(), info)
        record["status"] = "complete"
        print(f"完成：{output.resolve()}", flush=True)
    except BaseException as error:
        record.update(status="failed", error=str(error))
        if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
            print("顯存不足：請明確調低 --max-side 256 後重試；VACE 也可調低 --window-size 17。", file=sys.stderr)
        raise
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if device.type == "cuda":
            record["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        (output / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def entrypoint():
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    entrypoint()
