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


def to_model_input(rai: sitk.Image, hw: int, wl: float, ww: float) -> torch.Tensor:
    """RAI CT -> (1, Z, 1, hw, hw) windowed, in-plane-resized tensor."""
    arr = window_ct(sitk.GetArrayFromImage(rai).astype(np.float32), wl, ww)  # (Z, Y, X)
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(1)             # (Z, 1, Y, X)
    t = F.interpolate(t, size=(hw, hw), mode="bilinear", align_corners=False)
    return t.unsqueeze(0)                                                     # (1, Z, 1, hw, hw)
