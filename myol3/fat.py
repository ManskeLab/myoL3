"""Fat post-processing + body-composition quantification for an L3 muscle seg.

Each muscle is eroded to an interior "core" (the boundary rim reads as fat by
partial-volume averaging and is handled separately), and voxels are classified
by HU into four compartments:

    comp 1  muscle            core, HU >= LAMA_HI (default 30)
    comp 2  fat-PV muscle     core, FAT_HU <= HU < LAMA_HI  (low-attenuation)
    comp 3  intramuscular fat core, HU < FAT_HU (default -30)
    comp 4  outer-edge PV fat the eroded rim

Total muscle = mask minus intramuscular fat (comp 3) = comps 1+2+4.
Left/right split at the patient midline (x-centroid; RAI -> +x is patient Right).

Metrics per muscle x side (mm^2): muscle_csa (per-slice CSA, edge-outlier
trimmed), fat_pv_muscle_csa, intramuscular_fat_csa, outer_edge_pv_fat_csa —
each {mean, median, std, n_slices}.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from . import config

COMPARTMENTS = ["muscle_csa", "fat_pv_muscle_csa",
                "intramuscular_fat_csa", "outer_edge_pv_fat_csa"]


def _trim_edges(profile, k=3.0, rel_floor=0.10):
    nz = np.where(profile > 0)[0]
    if nz.size == 0:
        return np.array([])
    a = profile[nz.min():nz.max() + 1]
    n = a.size
    if n <= 4:
        return a
    lo, hi = n // 5, n - n // 5
    center = a[lo:hi] if hi > lo else a
    med = np.median(center)
    mad = 1.4826 * np.median(np.abs(center - med))
    scale = max(mad, rel_floor * med, 1e-6)
    out = lambda v: abs(v - med) > k * scale
    i = 0
    while i < n and out(a[i]):
        i += 1
    j = n - 1
    while j > i and out(a[j]):
        j -= 1
    kept = a[i:j + 1]
    return kept if kept.size else a


def _stats(areas):
    if len(areas) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "n_slices": 0}
    a = np.asarray(areas, float)
    return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "std": round(float(a.std()), 2), "n_slices": int(a.size)}


def _gmm2_crossover(x, default=30.0, lo=-200.0, hi=250.0, iters=60):
    """Per-muscle muscle / low-attenuation-muscle split: 2-component 1-D GMM
    (pure-numpy EM) on this muscle's core HU; returns the crossover HU between
    the two modes. Adapts to each scan's muscle attenuation. Falls back to
    `default` when the data isn't clearly bimodal / too sparse."""
    x = x[(x >= lo) & (x <= hi)].astype(np.float64)
    if x.size < 500:
        return default, False
    m = np.median(x)
    mu = np.array([x[x < m].mean(), x[x >= m].mean()])
    var = np.array([max(x[x < m].var(), 1.0), max(x[x >= m].var(), 1.0)])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        p0 = w[0] * np.exp(-(x - mu[0]) ** 2 / (2 * var[0])) / np.sqrt(2 * np.pi * var[0])
        p1 = w[1] * np.exp(-(x - mu[1]) ** 2 / (2 * var[1])) / np.sqrt(2 * np.pi * var[1])
        s = p0 + p1 + 1e-12
        r0, r1 = p0 / s, p1 / s
        n0, n1 = r0.sum(), r1.sum()
        if n0 < 1 or n1 < 1:
            return default, False
        mu = np.array([(r0 * x).sum() / n0, (r1 * x).sum() / n1])
        var = np.array([max((r0 * (x - mu[0]) ** 2).sum() / n0, 1.0),
                        max((r1 * (x - mu[1]) ** 2).sum() / n1, 1.0)])
        w = np.array([n0 / x.size, n1 / x.size])
    lo_i, hi_i = (0, 1) if mu[0] <= mu[1] else (1, 0)
    if mu[hi_i] - mu[lo_i] < 5:
        return default, False
    xs = np.linspace(mu[lo_i], mu[hi_i], 1000)
    d_lo = w[lo_i] * np.exp(-(xs - mu[lo_i]) ** 2 / (2 * var[lo_i])) / np.sqrt(2 * np.pi * var[lo_i])
    d_hi = w[hi_i] * np.exp(-(xs - mu[hi_i]) ** 2 / (2 * var[hi_i])) / np.sqrt(2 * np.pi * var[hi_i])
    return float(xs[np.argmin(np.abs(d_lo - d_hi))]), True


def quantify(ct: sitk.Image, seg: sitk.Image, fat_hu=None, lama_hi=None, erode=None):
    """(CT, muscle seg) -> (total_muscle_seg, composition_seg, metrics dict).

    composition_seg encodes muscle*10 + compartment (11..34).
    """
    fat_hu = config.FAT_HU if fat_hu is None else fat_hu
    if lama_hi is None:
        lama_hi = config.LAMA_HI          # may still be None -> adaptive per muscle
    erode = config.FAT_ERODE if erode is None else erode
    adaptive = lama_hi is None

    ct_a = sitk.GetArrayFromImage(ct).astype(np.float32)
    seg_a = sitk.GetArrayFromImage(seg)
    if ct_a.shape != seg_a.shape:
        raise ValueError(f"CT {ct_a.shape} and seg {seg_a.shape} grids differ")
    sx, sy, sz = seg.GetSpacing()
    px_area = sx * sy
    struct = np.zeros((3, 3, 3), bool); struct[1] = True

    fg = seg_a > 0
    midline = float(np.argwhere(fg)[:, 2].mean()) if fg.any() else ct_a.shape[2] / 2.0
    is_right = np.arange(ct_a.shape[2]) > midline

    total = seg_a.copy()
    comp = np.zeros_like(seg_a)
    metrics = {}
    thresholds = {}
    for lid, name in config.MUSCLE_LABELS.items():
        mask = seg_a == lid
        core = (ndimage.binary_erosion(mask, structure=struct, iterations=max(1, erode))
                if mask.any() else mask)
        rim = mask & ~core
        thr = _gmm2_crossover(ct_a[core])[0] if adaptive else lama_hi
        thresholds[name] = round(float(thr), 2)
        c3 = core & (ct_a < fat_hu)
        c2 = core & (ct_a >= fat_hu) & (ct_a < thr)
        c1 = core & (ct_a >= thr)
        c4 = rim
        total[c3] = 0
        comp[c1] = lid * 10 + 1; comp[c2] = lid * 10 + 2
        comp[c3] = lid * 10 + 3; comp[c4] = lid * 10 + 4
        total_muscle = mask & ~c3

        metrics[name] = {}
        for side, side_sel in (("L", ~is_right), ("R", is_right)):
            sidevol = np.zeros_like(mask); sidevol[:, :, side_sel] = True
            prof = (total_muscle & sidevol).sum(axis=(1, 2)).astype(np.float64) * px_area
            kept = _trim_edges(prof)
            nz = np.where(prof > 0)[0]
            zlo, zhi = (int(nz.min()), int(nz.max()) + 1) if nz.size else (0, 0)
            cp = lambda cm: (cm & sidevol).sum(axis=(1, 2)).astype(np.float64)[zlo:zhi] * px_area
            metrics[name][side] = {
                "muscle_csa": _stats(kept),
                "fat_pv_muscle_csa": _stats(cp(c2)),
                "intramuscular_fat_csa": _stats(cp(c3)),
                "outer_edge_pv_fat_csa": _stats(cp(c4)),
            }

    metrics["_thresholds_hu"] = {"fat": float(fat_hu),
                                 "muscle_low_atten_split": thresholds,
                                 "adaptive": bool(adaptive)}
    out_seg = sitk.GetImageFromArray(total.astype(np.uint8)); out_seg.CopyInformation(seg)
    out_comp = sitk.GetImageFromArray(comp.astype(np.uint8)); out_comp.CopyInformation(seg)
    return out_seg, out_comp, metrics
