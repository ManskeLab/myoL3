# myoL3

Automated **L3-level trunk-muscle analysis from CT**. Give it a full-body CT;
it localizes the L3 level and crops to it, segments the trunk muscles, strips
interior fat, and quantifies per-muscle, per-side body composition.

[Hugging Face](https://huggingface.co/YousifKhoury/myoL3) | [GitHub](https://github.com/Manskelab/myoL3)

---

## Pipeline

```
full-body CT  ->  L3 localizer (presence-gated sliding window)  ->  z-crop
              ->  muscle segmentation (nnU-Net, in-Python)       ->  raw labels
              ->  strip interior fat (IMAT)                       ->  TOTAL muscle seg
              ->  per-muscle / per-side composition metrics       ->  JSON
```

Muscles: psoas, quadratus lumborum, erector spinae / multifidus. The
muscle/low-attenuation split is a per-scan, per-muscle GMM crossover; interior
fat (HU < −30) is removed to form the total-muscle segmentation.

---

## Installation

Install PyTorch first ([pytorch.org](https://pytorch.org)), then:

```bash
pip install myoL3
```

Download the model weights (localizer + segmenter, once only):

```bash
myol3-install /path/to/models/dir
```

On shared HPC clusters, point at existing files instead:

```bash
export MYOL3_LOCALIZER_CKPT=/shared/models/myol3/l3_localizer.pt
export MYOL3_SEGMENTER_CKPT=/shared/models/myol3/muscle_seg.pth
```

---

## Usage

```bash
# full pipeline -> total muscle seg + composition map + metrics JSON
myol3 -i fullbody_ct.nii.gz -o total_muscle_seg.nii.gz \
      --save-comp composition.nii.gz --save-metrics metrics.json

# localize + crop only (no segmentation model needed)
myol3 -i fullbody_ct.nii.gz --save-crop l3_crop.nii.gz
```

| Flag | Description |
|---|---|
| `-i, --input` | full-body CT (`.nii` / `.nii.gz`) — required |
| `-o, --output` | total muscle segmentation (interior fat stripped); omit to only localize/crop |
| `--save-crop` | also write the L3-cropped CT |
| `--save-comp` | also write the 4-compartment map (`muscle*10 + compartment`) |
| `--save-metrics` | also write per-muscle / per-side metrics (`.json`) |
| `--localizer-ckpt` / `--segmenter-ckpt` | checkpoint overrides |
| `--pad` | extra slices each side of the L3 crop |
| `--device` | `cpu` \| `cuda` (auto if unset) |

Python API:

```python
import myol3
myol3.run("fullbody_ct.nii.gz", "total_muscle_seg.nii.gz",
          save_comp="composition.nii.gz", save_metrics="metrics.json")
```

---

## Metrics

Per muscle × side (L/R), in mm² (`mean`, `median`, `std`, `n_slices`):

- `muscle_csa` — total-muscle cross-sectional area (edge-outlier trimmed)
- `fat_pv_muscle_csa` — low-attenuation ("fat partial-volume") muscle
- `intramuscular_fat_csa` — solidly-fat interior (IMAT)
- `outer_edge_pv_fat_csa` — partial-volume boundary rim

---

## License

MIT
