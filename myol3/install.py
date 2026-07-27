"""`myol3-install <dir>` — download the two model checkpoints from Hugging Face
and record their paths so `myol3` finds them automatically.

Mirrors veinseg-install, extended to the two checkpoints myoL3 needs:
  - l3_localizer.pt  (L3 localizer)
  - muscle_seg.pth   (muscle segmentation, self-contained nnU-Net checkpoint)

Paths can also be overridden any time with:
  export MYOL3_LOCALIZER_CKPT=/path/to/l3_localizer.pt
  export MYOL3_SEGMENTER_CKPT=/path/to/muscle_seg.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import config


def _download(repo, fname, out_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fname, local_dir=str(out_dir))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: myol3-install <directory>")
        print()
        print("  Downloads the myoL3 checkpoints from Hugging Face")
        print(f"  ({config.LOCALIZER_HF_REPO}) into <directory> and records their")
        print("  paths so the `myol3` command finds them automatically:")
        print(f"    - {config.LOCALIZER_HF_FILE}   (L3 localizer)")
        print(f"    - {config.SEGMENTER_HF_FILE}   (muscle segmentation)")
        print()
        print("  Paths can be overridden any time with:")
        print("    export MYOL3_LOCALIZER_CKPT=/path/to/l3_localizer.pt")
        print("    export MYOL3_SEGMENTER_CKPT=/path/to/muscle_seg.pth")
        sys.exit(0)

    out = Path(sys.argv[1]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for kind in ("localizer", "segmenter"):
        repo, fname = config._HF[kind]
        if not repo:
            print(f"[myol3-install] skip {kind}: no HF repo configured in config.py")
            continue
        dest = out / fname
        if dest.exists():
            print(f"[myol3-install] {kind} already present: {dest}")
        else:
            print(f"[myol3-install] downloading {repo}/{fname} ...")
            _download(repo, fname, out)
            print(f"[myol3-install] -> {dest}")
        paths[kind] = str(dest)

    if paths:
        saved = config.save_checkpoint_paths(**paths)
        print()
        for k, v in paths.items():
            print(f"  {k:10}: {v}")
        print(f"  config    : {saved}")
        print()
        print("  Ready. Run:")
        print("    myol3 -i fullbody_ct.nii.gz -o total_muscle_seg.nii.gz")
    else:
        print("Nothing downloaded — configure the HF repo ids in config.py.")


if __name__ == "__main__":
    main()
