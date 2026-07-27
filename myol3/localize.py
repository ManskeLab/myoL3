"""L3 localizer: sliding-window inference over a full volume -> z-crop bounds."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch

from . import config
from .model import L3BoundaryLocalizer, bounds_from_boundaries
from .preprocess import to_model_input


class L3Localizer:
    def __init__(self, ckpt: str, device: str = "cpu"):
        self.device = device
        self.model = L3BoundaryLocalizer().to(device).eval()
        state = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(state["model"] if "model" in state else state)

    @torch.no_grad()
    def predict_bounds(self, rai: sitk.Image) -> tuple[int, int]:
        """Return (z_min, z_max) in `rai`'s z-index space.

        Sliding-window accumulation, weighted by presence (gates off chest/pelvis
        windows) and sharpness. Bounds are picked by _joint_bounds, which anchors
        on the presence peak and caps the span at LOCALIZER_MAX_MM — so a spurious
        boundary vote far from L3 can't stretch the crop (the CRIT_0152 failure).
        """
        x = to_model_input(rai, config.LOCALIZER_HW, config.WL, config.WW).to(self.device)
        z = x.shape[1]
        win, stride = config.LOCALIZER_WIN, config.LOCALIZER_STRIDE

        # span cap in slices from the (post-resample) z-spacing
        max_slices = None
        if config.LOCALIZER_MAX_MM and config.LOCALIZER_MAX_MM > 0:
            max_slices = int(round(config.LOCALIZER_MAX_MM / rai.GetSpacing()[2]))

        if z <= win:
            out = self.model(x)
            z0, z1 = bounds_from_boundaries(out["z_top"][0], out["z_bot"][0], z)
            if max_slices is not None and (z1 - z0) > max_slices:
                z1 = z0 + max_slices
            return z0, z1

        starts = list(range(0, z - win + 1, stride))
        if starts[-1] != z - win:
            starts.append(z - win)

        g_top = np.zeros(z); g_bot = np.zeros(z)
        g_pres = np.zeros(z); g_cnt = np.zeros(z)
        for s in starts:
            out = self.model(x[:, s:s + win])
            pt = out["p_top"][0].cpu().numpy()
            pb = out["p_bot"][0].cpu().numpy()
            wgt = float(pt.max() * pb.max())          # window sharpness
            if "presence" in out:                     # gate off windows without L3
                wgt *= float(torch.sigmoid(out["presence"][0]))
            g_top[s:s + win] += wgt * pt
            g_bot[s:s + win] += wgt * pb
            if "pres_slice" in out:
                g_pres[s:s + win] += torch.sigmoid(out["pres_slice"][0]).cpu().numpy()
                g_cnt[s:s + win] += 1.0

        pres = (g_pres / np.clip(g_cnt, 1.0, None)) if g_cnt.any() else None
        return _joint_bounds(g_top, g_bot, max_slices, presence=pres)

    def crop(self, rai: sitk.Image, pad: int = 0) -> tuple[sitk.Image, tuple[int, int]]:
        z0, z1 = self.predict_bounds(rai)
        z0 = max(0, z0 - pad)
        z1 = min(rai.GetSize()[2] - 1, z1 + pad)
        return rai[:, :, z0:z1 + 1], (z0, z1)


def _joint_bounds(g_top, g_bot, max_slices, presence=None):
    """Ordered (z0, z1) with the span capped at max_slices.

    Presence is the reliable signal (trained ~0 off L3), so when available we
    anchor on it: z0 = strongest g_top below the presence peak, z1 = strongest
    g_bot above it, both within max_slices. A tall spurious boundary vote outside
    the L3 region cannot be selected. Falls back to the raw peaks when the two
    are already ordered and within the cap, else anchors on the taller peak.
    """
    n = len(g_top)
    ai = int(g_top.argmax()); bi = int(g_bot.argmax())
    if presence is not None and max_slices is not None and max_slices < n:
        ms = int(max_slices)
        c = int(np.argmax(presence))
        lo = max(0, c - ms)
        z0 = lo + int(np.argmax(g_top[lo:c + 1]))
        hi = min(n, c + ms + 1)
        z1 = c + int(np.argmax(g_bot[c:hi]))
        if z1 - z0 > ms:
            z1 = z0 + ms
        return z0, z1
    if max_slices is None or max_slices >= n or 0 <= (bi - ai) <= max_slices:
        return tuple(sorted((ai, bi)))
    ms = int(max_slices)
    if float(g_top.max()) >= float(g_bot.max()):
        z0 = ai; z1 = z0 + int(np.argmax(g_bot[z0:min(n, z0 + ms + 1)]))
    else:
        z1 = bi; lo = max(0, z1 - ms); z0 = lo + int(np.argmax(g_top[lo:z1 + 1]))
    return z0, z1
