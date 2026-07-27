# myoL3

Automated **L3-level trunk-muscle analysis from CT**. The tool takes a full-body abdominal CT,
it resamples to the working resolution, localizes the L3 level and crops to it,
then runs muscle segmentation on the crop.

```
full-body CT (.nii.gz)
      │  resample to target spacing
      ▼
  L3 localizer (presence-gated sliding window)  ──►  z-crop
      ▼
  muscle segmentation (nnU-Net, in-Python)  ──►  raw label map
      ▼
  strip interior fat (intramuscular adipose tissue)  ──►  TOTAL muscle segmentation (final)
```

The muscle segmenter downloads weights from HuggingFace and runs
inference on the full CT (Computed Tomography) first to get L3 bounds.
The segmentation model is then run on the cropped L3-bounded image to label the muscles. The fat step removes solidly-fat interior voxels (< −30 Hounsfield units, HU, inside the eroded muscle core) from each muscle.

## Example (L3 axial slice)

| Input CT | Muscle segmentation | Body-composition map |
|---|---|---|
| ![L3 axial CT](assets/axial_abd_slice.png) | ![muscle segmentation](assets/axial_abd_slice_segmented.png) | ![fat composition](assets/axial_abd_slice_fat.png) |

- **Muscle segmentation** — psoas (red), quadratus lumborum (green), erector
  spinae / multifidus (blue).
- **Body-composition map** — each muscle is split into four compartments: muscle,
  low-attenuation ("fat partial-volume") muscle, intramuscular fat, and the
  partial-volume boundary rim. The muscle / low-attenuation split is a per-scan,
  per-muscle Gaussian Mixture Model (GMM), interior fat (< −30 HU) is
  removed to form the total-muscle segmentation.

  > **Gaussian Mixture Model (GMM)** — a muscle's HU values form two overlapping populations,
  > denser normal muscle and lower-HU fatty muscle. The GMM fits one bell curve
  > to each and puts the split where they cross, so the threshold adapts to each
  > patient's muscle attenuation instead of a fixed cutoff (a myosteatotic
  > patient's split lands much lower).

## Installation

Install PyTorch first ([pytorch.org](https://pytorch.org)), then:

```bash
pip install myoL3
```

Or clone and install editable for development:

```bash
git clone https://github.com/Manskelab/myoL3 myoL3
cd myoL3
pip install -e .
```

## Model weights

Both checkpoints live in one Hugging Face repo
([`YousifKhoury/myoL3`](https://huggingface.co/YousifKhoury/myoL3)):
`l3_localizer.pt` (L3 bounds localizer) and `muscle_seg.pth` (UNet muscle
segmenter). Download them once:

```bash
myol3-install /path/to/models        # downloads both + remembers the paths
```

or point at local files without downloading:

```bash
export MYOL3_LOCALIZER_CKPT=/path/to/l3_localizer.pt
export MYOL3_SEGMENTER_CKPT=/path/to/muscle_seg.pth
```

Resolution order per model: `--*-ckpt` flag → env var → `myol3-install` path
(`~/.myol3/config.json`) → Hugging Face download.

## Models

- **L3 localizer** (`l3_localizer.pt`) — a Residual Network (ResNet-18) encodes
  each axial slice, a bidirectional Gated Recurrent Unit (GRU) runs over the
  superior–inferior axis, a soft-argmax head regresses the top and bottom L3
  slice, and a presence head gates out slices that contain no L3. Applied as a
  sliding window over the full volume.
- **Muscle segmenter** (`muscle_seg.pth`) — a 3D UNet (no-new-UNet / nnU-Net,
  full-resolution configuration) run with sliding-window inference, loaded and
  driven directly in Python.

## Usage

```bash
# full pipeline: localize L3 -> crop -> segment -> strip fat -> total muscle seg
myol3 -i fullbody_ct.nii.gz -o total_muscle_seg.nii.gz

# also save the crop, the 4-compartment map, and per-muscle/side metrics
myol3 -i fullbody_ct.nii.gz -o total_muscle_seg.nii.gz \
      --save-crop l3_crop.nii.gz --save-comp composition.nii.gz --save-metrics metrics.json

# localize + crop only (no segmentation model needed)
myol3 -i fullbody_ct.nii.gz --save-crop l3_crop.nii.gz

# overrides
myol3 -i ct.nii.gz -o seg.nii.gz \
      --localizer-ckpt /path/l3_localizer.pt \
      --segmenter-ckpt /path/muscle_seg.pth \
      --pad 2 --device cuda
```

| Flag | Meaning |
|---|---|
| `-i, --input` | full-body CT (required) |
| `-o, --output` | **total** muscle seg (fat stripped); omit to only localize/crop |
| `--save-crop` | also write the L3-cropped CT |
| `--save-comp` | also write the 4-compartment map (`muscle*10 + compartment`) |
| `--save-metrics` | also write per-muscle/side metrics (`.json`) |
| `--localizer-ckpt` / `--segmenter-ckpt` | checkpoint overrides |
| `--pad` | extra slices each side of the L3 crop |
| `--device` | `cpu` or `cuda` (auto if unset) |

Python API:

```python
import myol3
myol3.run("fullbody_ct.nii.gz", "total_muscle_seg.nii.gz",
          save_crop="l3_crop.nii.gz", save_metrics="metrics.json")
```

## Configuration

`myol3/config.py`:
- `TARGET_SPACING` — spacing (mm) to resample the input to; set to what the models were trained at (`None` = keep native).
- `LOCALIZER_HW / LOCALIZER_WIN / LOCALIZER_STRIDE` — localizer input size and sliding-window params.
- `LOCALIZER_MAX_MM` — hard cap on the L3 crop length (mm) so a spurious vote can't stretch it.
- `WL / WW` — CT window for the localizer.
- `FAT_HU / FAT_ERODE` — intramuscular-fat threshold and interior-core erosion for fat stripping.
- `LAMA_HI` — muscle / low-attenuation split HU; `None` = per-scan, per-muscle Gaussian Mixture Model (GMM) (adaptive).
- `SEG_TILE_STEP / SEG_USE_MIRRORING` — nnU-Net sliding-window step and test-time mirroring.
