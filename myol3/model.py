"""L3 slice-localizer (boundary / soft-argmax head) — inference architecture.

Mirrors the trained model: a per-slice ResNet-18 encoder feeds a BiGRU over z,
and a soft-argmax head emits the two L3 boundary positions (z_min, z_max).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision


class SliceEncoder(nn.Module):
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        old = net.conv1                       # 1-channel CT input
        net.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size,
                              old.stride, old.padding, bias=False)
        self.trunk = nn.Sequential(*list(net.children())[:-1])   # -> (N, 512, 1, 1)
        self.proj = nn.Linear(512, feat_dim)

    def forward(self, x):                     # (N, 1, H, W)
        return self.proj(self.trunk(x).flatten(1))


class L3BoundaryLocalizer(nn.Module):
    """Input (B, Z, 1, H, W); output dict with per-z boundary distributions and
    the soft-argmax boundary positions z_top, z_bot."""

    def __init__(self, feat_dim: int = 256, rnn_hidden: int = 128,
                 rnn_layers: int = 2, encode_chunk: int = 64):
        super().__init__()
        self.encoder = SliceEncoder(feat_dim)
        self.encode_chunk = encode_chunk
        self.rnn = nn.GRU(feat_dim, rnn_hidden, rnn_layers, batch_first=True,
                          bidirectional=True, dropout=0.1 if rnn_layers > 1 else 0.0)
        self.head = nn.Linear(2 * rnn_hidden, 2)
        self.presence_head = nn.Linear(2 * rnn_hidden, 1)  # P(slice is L3), per slice

    def _encode(self, flat):
        if self.encode_chunk and self.encode_chunk < flat.shape[0]:
            return torch.cat([self.encoder(c)
                              for c in flat.split(self.encode_chunk, dim=0)], dim=0)
        return self.encoder(flat)

    def forward(self, x, mask=None):
        b, z, c, h, w = x.shape
        feats = self._encode(x.reshape(b * z, c, h, w)).reshape(b, z, -1)
        seq, _ = self.rnn(feats)
        logits = self.head(seq)
        top, bot = logits[..., 0], logits[..., 1]
        pres_slice = self.presence_head(seq).squeeze(-1)   # (B, Z) presence logit
        if mask is not None:
            fill = torch.finfo(top.dtype).min
            top = top.masked_fill(mask == 0, fill)
            bot = bot.masked_fill(mask == 0, fill)
        p_top, p_bot = torch.softmax(top, 1), torch.softmax(bot, 1)
        idx = torch.arange(z, device=x.device, dtype=p_top.dtype)
        pres_pool = pres_slice if mask is None else \
            pres_slice.masked_fill(mask == 0, torch.finfo(pres_slice.dtype).min)
        return {"p_top": p_top, "p_bot": p_bot,
                "z_top": (p_top * idx).sum(1), "z_bot": (p_bot * idx).sum(1),
                "pres_slice": pres_slice, "presence": pres_pool.max(dim=1).values}


def window_ct(arr: np.ndarray, wl: float = 50.0, ww: float = 400.0) -> np.ndarray:
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def bounds_from_boundaries(z_top, z_bot, n: int) -> tuple[int, int]:
    a, b = sorted((int(round(float(z_top))), int(round(float(z_bot)))))
    return max(0, min(a, n - 1)), max(0, min(b, n - 1))
