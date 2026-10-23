"""Wan VACE inference; original missing pixels never enter the model."""
import gc
import hashlib
import os
from pathlib import Path
import numpy as np
import torch

from core import add_prediction, condition_window, windows


def reconstruct(frames, mask, sums, counts, dimensions, args):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
    from huggingface_hub import model_info
    from transformers import UMT5EncoderModel

    model_id = f"Wan-AI/Wan2.1-VACE-{args.model_size.upper()}-diffusers"
    # Official names use 1.3B / 14B; resolve once so all files share one revision.
    revision = model_info(model_id).sha
    device = torch.device(args.device)
    # Turing (2080 Ti) has no native BF16. FP32 avoids FP16 overflows in Wan.
    dtype = (torch.bfloat16 if device.type == "cuda" and
             torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
             else torch.float32)
    print(f"載入 {model_id}（{dtype}，首次使用會下載權重）", flush=True)
    prompt = "A continuous video with natural, temporally consistent motion."
    negative_prompt = "flicker, distorted objects, abrupt motion, scene cuts"
    cache = Path(os.environ.get("FRAME_FILL_CACHE", Path.home() / ".cache" / "video-frame-inpainting-study"))
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((prompt + "\n" + negative_prompt).encode()).hexdigest()[:16]
    prompt_file = cache / f"vace-prompt-{revision}-{digest}.pt"
    encoder = None
    if not prompt_file.is_file():
        encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder",
                    revision=revision, torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", revision=revision,
                                          torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, revision=revision, vae=vae,
                                           text_encoder=encoder, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.vae.enable_tiling()
    # Encode fixed, content-neutral prompts on CPU; the XXL encoder does not fit
    # on an 11 GB GPU. Do not generate prompts from the held-out source frames.
    if prompt_file.is_file():
        embeddings = torch.load(prompt_file, map_location="cpu", weights_only=True)
        positive, negative = embeddings["positive"], embeddings["negative"]
    else:
        with torch.inference_mode():
            positive, negative = pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt,
                device=torch.device("cpu"), dtype=torch.float32)
        temporary = prompt_file.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"positive": positive, "negative": negative}, temporary)
        temporary.replace(prompt_file)
    pipe.register_modules(text_encoder=None)
    del encoder
    gc.collect()
    if device.type == "cuda":
        pipe.enable_model_cpu_offload(gpu_id=device.index or 0)
    else:
        pipe.to(device)
    positive = positive.to(device=device, dtype=dtype)
    negative = negative.to(device=device, dtype=dtype)
    with torch.inference_mode():
        for number, (start, end) in enumerate(windows(len(frames), args.window_size)):
            if not mask[start:end].any():
                continue
            print(f"VACE 視窗 {start + 1}–{end}", flush=True)
            video, masks = condition_window(frames, mask, start, end, args.window_size, dimensions)
            prediction = pipe(video=video, mask=masks, prompt_embeds=positive,
                              negative_prompt_embeds=negative,
                              width=dimensions[0], height=dimensions[1],
                              num_frames=args.window_size, num_inference_steps=args.steps,
                              guidance_scale=5.0, max_sequence_length=226,
                              generator=torch.Generator(device="cpu").manual_seed(args.seed + number),
                              output_type="np").frames[0]
            for local, index in enumerate(range(start, end)):
                if mask[index]:
                    add_prediction(sums, counts, index, prediction[local])
    return {"id": model_id, "revision": revision, "dtype": str(dtype),
            "cpu_offload": device.type == "cuda", "overlap_frames": 8,
            "prompt": prompt, "negative_prompt": negative_prompt}
