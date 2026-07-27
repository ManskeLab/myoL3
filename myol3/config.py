"""Central configuration: checkpoint resolution + processing parameters.

Checkpoints are resolved (per model) in this order:
  1. explicit path passed on the CLI / API
  2. env var  MYOL3_LOCALIZER_CKPT / MYOL3_SEGMENTER_CKPT
  3. path saved by `myol3-install`  (~/.myol3/config.json)
  4. download from Hugging Face  (repo ids below)

Fill in the Hugging Face repo ids once the models are published.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ── Hugging Face model locations ─────────────────────────────────────────────
# Both checkpoints live in one repo: https://huggingface.co/YousifKhoury/myoL3
LOCALIZER_HF_REPO = "YousifKhoury/myoL3"
LOCALIZER_HF_FILE = "l3_localizer.pt"
SEGMENTER_HF_REPO = "YousifKhoury/myoL3"
SEGMENTER_HF_FILE = "muscle_seg.pth"

# ── Processing parameters (set to what the models were trained at) ───────────
TARGET_SPACING = None         # (sx, sy, sz) mm to resample the input to, or None = native
LOCALIZER_HW = 192            # in-plane size the localizer sees
LOCALIZER_WIN = 96            # sliding-window length (slices)
LOCALIZER_STRIDE = 24         # sliding-window stride (slices)
LOCALIZER_MAX_MM = 43.75      # hard cap on the L3 crop length in mm (0 = off)
WL, WW = 50.0, 400.0          # CT window level / width for the localizer

# ── Fat post-processing (strip interior IMAT from the muscle) ────────────────
FAT_HU = -30.0                # HU below this (inside the core) is fat/IMAT
LAMA_HI = None                # muscle/low-atten split: None = per-image, per-muscle
                              # GMM crossover (adaptive); a number forces a fixed HU
FAT_ERODE = 2                 # in-plane erosion (voxels) to define the interior core
MUSCLE_LABELS = {1: "psoas", 2: "quadratus_lumborum", 3: "erector_spinae"}

# ── Muscle segmenter nnU-Net inference ───────────────────────────────────────
SEG_TILE_STEP = 0.5           # sliding-window step (fraction of patch), nnU-Net default
SEG_USE_MIRRORING = True      # test-time augmentation (mirroring)

# ── Where installed-checkpoint paths are remembered ─────────────────────────
CONFIG_PATH = Path(os.environ.get("MYOL3_HOME", Path.home() / ".myol3")) / "config.json"

_HF = {
    "localizer": (LOCALIZER_HF_REPO, LOCALIZER_HF_FILE),
    "segmenter": (SEGMENTER_HF_REPO, SEGMENTER_HF_FILE),
}


def _saved() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_checkpoint_paths(localizer: str | None = None, segmenter: str | None = None) -> Path:
    cfg = _saved()
    if localizer:
        cfg["localizer"] = str(localizer)
    if segmenter:
        cfg["segmenter"] = str(segmenter)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return CONFIG_PATH


def resolve_checkpoint(kind: str, explicit: str | None = None) -> str:
    """kind is 'localizer' or 'segmenter'."""
    if explicit:
        return str(explicit)
    env = os.environ.get(f"MYOL3_{kind.upper()}_CKPT")
    if env:
        return env
    saved = _saved().get(kind)
    if saved:
        return saved
    repo, fname = _HF[kind]
    if not repo:
        raise FileNotFoundError(
            f"No {kind} checkpoint configured. Do one of:\n"
            f"  - pass --{kind}-ckpt <path>\n"
            f"  - export MYOL3_{kind.upper()}_CKPT=<path>\n"
            f"  - run `myol3-install <dir>` after setting {kind.upper()}_HF_REPO in config.py"
        )
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fname)
