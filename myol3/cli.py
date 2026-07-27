"""`myol3` command-line entry point."""
from __future__ import annotations

import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(
        prog="myol3",
        description="Localize L3 in a full-body CT, crop, and segment trunk muscle.")
    ap.add_argument("-i", "--input", required=True, help="full-body CT (.nii/.nii.gz)")
    ap.add_argument("-o", "--output", help="output muscle segmentation (.nii.gz); "
                                           "omit to only localize+crop")
    ap.add_argument("--save-crop", help="also write the L3-cropped CT here")
    ap.add_argument("--save-comp", help="also write the 4-compartment map (muscle*10+comp)")
    ap.add_argument("--save-metrics", help="also write full per-muscle/side metrics (.json)")
    ap.add_argument("--localizer-ckpt", help="override L3 localizer checkpoint")
    ap.add_argument("--segmenter-ckpt", help="override muscle segmentation checkpoint")
    ap.add_argument("--pad", type=int, default=0, help="extra slices each side of the L3 crop")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = ap.parse_args()

    run(input_path=args.input, output_path=args.output,
        localizer_ckpt=args.localizer_ckpt, segmenter_ckpt=args.segmenter_ckpt,
        device=args.device, pad=args.pad, save_crop=args.save_crop,
        save_comp=args.save_comp, save_metrics=args.save_metrics)


if __name__ == "__main__":
    main()
