"""I/O + resampling + model-input preparation (all in RAI so axial = z)."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from .model import window_ct

ORIENT = "RAI"


def load_rai(path) -> sitk.Image:
    return sitk.DICOMOrient(sitk.ReadImage(str(path)), ORIENT)


def write(img: sitk.Image, path) -> None:
    sitk.WriteImage(img, str(path), useCompression=True)


def resample_to_spacing(img: sitk.Image, spacing, is_label: bool = False) -> sitk.Image:
    """Resample to an isotropic-ish target spacing (sx, sy, sz) mm; None = no-op."""
    if spacing is None:
        return img
    in_sp = img.GetSpacing()
    in_sz = img.GetSize()
    out_sz = [int(round(in_sz[i] * in_sp[i] / spacing[i])) for i in range(3)]
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(img, out_sz, sitk.Transform(), interp, img.GetOrigin(),
                         spacing, img.GetDirection(), 0, img.GetPixelID())


def to_model_input(rai: sitk.Image, hw: int, wl: float, ww: float,
                   chunk: int = 64) -> torch.Tensor:
    """RAI CT -> (1, Z, 1, hw, hw) windowed, in-plane-resized tensor.

    Windowed + resized to hw×hw in z-chunks so we never hold a full-resolution
    float32 copy of the whole volume in memory — only the int16 array plus one
    chunk at a time. The output is small (Z×hw×hw)."""
    arr = sitk.GetArrayFromImage(rai)                 # (Z, Y, X) int16, single copy
    z = arr.shape[0]
    out = []
    for s in range(0, z, chunk):
        block = window_ct(arr[s:s + chunk], wl, ww)   # small float32 block (b, Y, X)
        t = torch.from_numpy(np.ascontiguousarray(block)).unsqueeze(1)   # (b, 1, Y, X)
        t = F.interpolate(t, size=(hw, hw), mode="bilinear", align_corners=False)
        out.append(t)                                 # (b, 1, hw, hw)  — tiny
    return torch.cat(out, dim=0).unsqueeze(0)         # (1, Z, 1, hw, hw)
