"""Official LDMVFI midpoint checkpoint, in its separate legacy environment."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from .core import add_prediction

LDMVFI_REVISION = "eee2dc3566f21f22787d66b0f0337e48318403c6"
LDMVFI_WEIGHT_ID = "1_Xx2fBYQT9O-6O3zjzX76O9XduGnCh_7"


def midpoint_conditions(frames, mask, index, dimensions):
    """Read only the two retained neighbours; tolerate narrow portrait videos."""
    if not (0 < index < len(frames) - 1 and mask[index]
            and not mask[index - 1] and not mask[index + 1]):
        raise ValueError("LDMVFI 僅支援兩側均為保留幀的單幀缺口。")
    result = {}
    for name, position in (("prev_frame", index - 1), ("next_frame", index + 1)):
        image = np.array(Image.fromarray(frames[position]).resize(dimensions, Image.Resampling.LANCZOS))
        # Upstream reflect-padding fails on sides shorter than 256. NumPy's
        # repeated reflection also handles very narrow inputs without copying GT.
        result[name] = np.pad(image, ((0, max(0, 256 - dimensions[1])),
                                     (0, max(0, 256 - dimensions[0])), (0, 0)), mode="reflect")
    return result


def reconstruct(frames, mask, sums, counts, dimensions, args):
    import torch

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("LDMVFI 的自訂運算需要 CUDA GPU；CPU 請改用 --model rife。")
    if torch.cuda.get_device_capability(device)[0] >= 12:
        raise ValueError("目前 LDMVFI 的舊版環境不支援 RTX 5090；請用已測試的 2080 環境，5090 可先用 RIFE。")
    try:
        import cupy
        import gdown
        from omegaconf import OmegaConf
        import pytorch_lightning  # Check dependencies before downloading weights.
    except ImportError as error:
        raise RuntimeError("請先執行 bash setup-ldmvfi.sh 並啟用 .venv-ldmvfi；參閱 README。") from error

    root = Path(os.environ.get("FRAME_FILL_CACHE", Path.home() / ".cache" / "video-frame-inpainting-study"))
    root.mkdir(parents=True, exist_ok=True)
    repo = root / f"LDMVFI-{LDMVFI_REVISION}"
    if not repo.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/danier97/LDMVFI.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", LDMVFI_REVISION],
                   check=True, stdout=subprocess.DEVNULL)
    checkpoint = root / "ldmvfi.ckpt"
    if not checkpoint.is_file():
        temporary = checkpoint.with_suffix(".download")
        if gdown.download(id=LDMVFI_WEIGHT_ID, output=str(temporary), quiet=False) is None:
            raise RuntimeError("無法下載官方 LDMVFI 權重，請確認 Google Drive 連線。")
        temporary.replace(checkpoint)

    # Both the official DDIM sampler and CuPy stream refer to the current GPU.
    torch.cuda.set_device(device)
    cupy.cuda.Device(device.index).use()
    sys.path.insert(0, str(repo))
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler

    config = OmegaConf.load(repo / "configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml")
    model = instantiate_from_config(config.model)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.to(device).eval()
    sampler = DDIMSampler(model)
    with torch.no_grad(), model.ema_scope():
        for index in np.flatnonzero(mask).tolist():
            torch.manual_seed(args.seed + index)
            np.random.seed((args.seed + index) % (2 ** 32))
            arrays = midpoint_conditions(frames, mask, index, dimensions)
            conditions = {name: torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                          / 127.5 - 1 for name, image in arrays.items()}
            condition, previous_features, next_features = model.get_learned_conditioning(conditions)
            latent, _ = sampler.sample(S=args.steps, conditioning=condition, batch_size=1,
                                      shape=(model.channels, *condition.shape[2:]), eta=1.0, verbose=False)
            decoded = model.decode_first_stage(latent, conditions, previous_features, next_features)
            prediction = decoded[0, :, :dimensions[1], :dimensions[0]].permute(1, 2, 0).cpu().numpy()
            add_prediction(sums, counts, index, (prediction + 1) / 2)
            print(f"LDMVFI 補幀 {index + 1}/{len(frames)}", flush=True)
    checksum = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return {"id": "danier97/LDMVFI", "revision": LDMVFI_REVISION,
            "weights_google_drive_id": LDMVFI_WEIGHT_ID, "weights_sha256": checksum.hexdigest(),
            "dtype": "torch.float32", "sampler": "DDIM", "eta": 1.0, "ema": True,
            "padding": "reflect-pad sides to at least 256, then official internal padding"}
