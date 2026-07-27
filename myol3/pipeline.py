"""End-to-end pipeline: full-body CT -> resample -> L3 localize/crop -> segment."""
from __future__ import annotations

from pathlib import Path

import torch

from . import config
from .preprocess import load_rai, resample_to_spacing, write


def run(input_path, output_path, localizer_ckpt=None, segmenter_ckpt=None,
        device=None, pad=0, save_crop=None, save_comp=None, save_metrics=None):
    """Segment L3-level muscle from a full-body CT.

    input_path  : full-body CT (.nii/.nii.gz)
    output_path : TOTAL muscle segmentation (interior fat stripped); None to skip
    save_crop   : optional path to also write the L3-cropped CT
    save_comp   : optional path to write the 4-compartment map (muscle*10+comp)
    save_metrics: optional path to write full per-muscle/side metrics (.json)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load + resample to the models' working resolution
    rai = load_rai(input_path)
    rai = resample_to_spacing(rai, config.TARGET_SPACING)

    # 2) localize L3 and crop in z
    from .localize import L3Localizer
    loc = L3Localizer(config.resolve_checkpoint("localizer", localizer_ckpt), device)
    crop, (z0, z1) = loc.crop(rai, pad=pad)
    print(f"L3 crop: z {z0}-{z1} ({z1 - z0 + 1} slices), size {crop.GetSize()}")
    if save_crop:
        write(crop, save_crop)
        print(f"  wrote crop -> {save_crop}")

    if output_path is None:
        return {"z_min": z0, "z_max": z1, "crop": save_crop}

    # 3) segment muscle on the crop
    from .segment import MuscleSegmenter
    seg = MuscleSegmenter(config.resolve_checkpoint("segmenter", segmenter_ckpt), device)
    raw_label = seg.predict(crop)

    # 4) strip interior fat + quantify -> total muscle seg (final) + metrics
    from .fat import quantify
    total_label, comp_label, metrics = quantify(crop, raw_label)
    for muscle in config.MUSCLE_LABELS.values():
        for side in ("L", "R"):
            s = metrics[muscle][side]
            print(f"  {muscle:20} {side}  muscle {s['muscle_csa']['mean']:7.1f}mm² "
                  f"(med {s['muscle_csa']['median']:.0f}, n {s['muscle_csa']['n_slices']})  "
                  f"IMfat {s['intramuscular_fat_csa']['mean']:5.1f}  "
                  f"fatPVmusc {s['fat_pv_muscle_csa']['mean']:6.1f}  "
                  f"edgePVfat {s['outer_edge_pv_fat_csa']['mean']:6.1f}")
    write(total_label, output_path)
    print(f"  wrote total muscle seg -> {output_path}")
    if save_comp:
        write(comp_label, save_comp)
        print(f"  wrote composition seg -> {save_comp}")
    if save_metrics:
        import json
        from pathlib import Path as _P
        _P(save_metrics).parent.mkdir(parents=True, exist_ok=True)
        _P(save_metrics).write_text(json.dumps(metrics, indent=2))
        print(f"  wrote metrics -> {save_metrics}")
    return {"z_min": z0, "z_max": z1, "crop": save_crop,
            "segmentation": str(output_path), "metrics": metrics}
