"""Pinned official RIFE_m code and weights; arbitrary-timestep interpolation."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from core import add_prediction

RIFE_REVISION = "5d8adbdd40e12c2c8f91930eff838aebe561c086"
RIFE_WEIGHT_ID = "147XVsDXBfJPlyct2jfo9kpbL944mNeZr"


def load_model(device):
    import gdown

    root = Path(os.environ.get("FRAME_FILL_CACHE", Path.home() / ".cache" / "video-frame-inpainting-study"))
    root.mkdir(parents=True, exist_ok=True)
    repo = root / f"RIFE-{RIFE_REVISION}"
    if not repo.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/hzwer/ECCV2022-RIFE.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", RIFE_REVISION], check=True,
                   stdout=subprocess.DEVNULL)
    weights = root / "RIFE_m.zip"
    checkpoint = root / "RIFE_m.flownet.pkl"
    if not checkpoint.is_file():
        if not weights.is_file():
            temporary = weights.with_suffix(".download")
            result = gdown.download(id=RIFE_WEIGHT_ID, output=str(temporary), quiet=False)
            if result is None:
                raise RuntimeError("無法下載官方 RIFE_m 權重，請確認 Google Drive 連線。")
            temporary.replace(weights)
        # Extract just the checkpoint, without trusting archive paths.
        with zipfile.ZipFile(weights) as archive:
            members = [n for n in archive.namelist() if n.endswith("flownet.pkl")]
            if len(members) != 1:
                raise ValueError("RIFE_m 權重壓縮檔內容不符預期。")
            temporary = checkpoint.with_suffix(".download")
            with archive.open(members[0]) as source, temporary.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
            temporary.replace(checkpoint)
    sys.path.insert(0, str(repo))
    network = importlib.import_module("model.IFNet_m").IFNet_m()
    # The upstream warper uses a module-level device; set it explicitly so
    # --device cpu and cuda:N do not accidentally use the default GPU.
    warper = importlib.import_module("model.warplayer")
    warper.device = device
    warper.backwarp_tenGrid.clear()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    network.load_state_dict(state, strict=True)
    network.to(device).eval()
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return network, checksum


def reconstruct(frames, mask, sums, counts, dimensions, args):
    device = torch.device(args.device)
    network, checksum = load_model(device)

    def tensor(index):
        image = Image.fromarray(frames[index]).resize(dimensions, Image.Resampling.LANCZOS)
        # RIFE's original cv2 input uses BGR, not RGB.
        array = np.array(image)[:, :, ::-1].copy()
        value = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
        return F.pad(value, (0, (-dimensions[0]) % 32, 0, (-dimensions[1]) % 32), mode="replicate")

    kept = np.flatnonzero(~mask)
    with torch.inference_mode():
        for left, right in zip(kept[:-1], kept[1:]):
            if right - left <= 1:
                continue
            first, last = tensor(left), tensor(right)
            print(f"RIFE 補幀 {left + 2}–{right}（端點 {left + 1}、{right + 1}）", flush=True)
            for index in range(left + 1, right):
                time = float((index - left) / (right - left))
                prediction = network(torch.cat((first, last), 1), timestep=time)[2][2]
                prediction = prediction[0, :, :dimensions[1], :dimensions[0]]
                rgb = prediction.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                add_prediction(sums, counts, index, rgb)
    return {"id": "hzwer/ECCV2022-RIFE RIFE_m", "revision": RIFE_REVISION,
            "weights_google_drive_id": RIFE_WEIGHT_ID, "weights_sha256": checksum,
            "dtype": "torch.float32"}
