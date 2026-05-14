"""USCorUNet model definition.

USCorUNet estimates dense bidirectional deformation fields between adjacent
ultrasound frames. The network combines:

- a ResUNet-style context encoder-decoder operating on
  ``[I0, I1, I1 - I0, grad(I0), grad(I1)]``;
- a shared-weight correlation encoder for local cost volumes at 1/8 scale.

Outputs:
    F01: ``[B, 2, H, W]`` flow from t0 to t1, defined on the I0 grid.
    F10: ``[B, 2, H, W]`` flow from t1 to t0, defined on the I1 grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_gn_groups(channels: int) -> int:
    """Choose a GroupNorm group count that divides ``channels``."""
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    """Convolution followed by GroupNorm and LeakyReLU."""

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(cout), cout, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResBlockGN(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(channels), channels, eps=1e-5, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(channels), channels, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + residual)


class DownGN(nn.Module):
    """Strided downsampling block."""

    def __init__(self, cin: int, cout: int, num_res: int = 2):
        super().__init__()
        self.down = ConvGNAct(cin, cout, k=3, s=2, p=1)
        self.res = nn.Sequential(*[ResBlockGN(cout) for _ in range(int(num_res))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpGN(nn.Module):
    """Upsampling block with skip fusion."""

    def __init__(self, cin: int, cout: int, num_res: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(cout), cout, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.fuse = nn.Conv2d(cout * 2, cout, 3, padding=1, bias=False)
        self.fuse_gn = nn.GroupNorm(_choose_gn_groups(cout), cout, eps=1e-5, affine=True)
        self.res = nn.Sequential(*[ResBlockGN(cout) for _ in range(int(num_res))])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.fuse_gn(self.fuse(x)))
        return self.res(x)


def local_correlation(f0: torch.Tensor, f1: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Compute a local cost volume between feature maps at the same resolution."""
    if f0.ndim != 4 or f1.ndim != 4:
        raise ValueError(f"Expected 4D feature maps, got {tuple(f0.shape)} and {tuple(f1.shape)}")
    if f0.shape != f1.shape:
        raise ValueError(f"Feature shape mismatch: {tuple(f0.shape)} vs {tuple(f1.shape)}")

    batch, channels, height, width = f0.shape
    radius = int(radius)
    kernel = 2 * radius + 1

    f1_unfolded = F.unfold(f1, kernel_size=kernel, padding=radius)
    f1_unfolded = f1_unfolded.view(batch, channels, kernel * kernel, height, width)
    corr = (f0.unsqueeze(2) * f1_unfolded).sum(dim=1)
    return corr / math.sqrt(float(channels) + 1e-6)


class CorrEncoder(nn.Module):
    """Shared-weight encoder producing 1/8-scale correlation features."""

    def __init__(self, in_ch: int = 2, base_ch: int = 24, num_res: int = 1):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, c1, 3, 1, 1),
            ResBlockGN(c1),
        )
        self.down1 = DownGN(c1, c2, num_res=num_res)
        self.down2 = DownGN(c2, c3, num_res=num_res)
        self.down3 = DownGN(c3, c3, num_res=num_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        return self.down3(x)


class USCorUNet(nn.Module):
    """Correlation-guided ResUNet for ultrasound deformation estimation."""

    def __init__(
        self,
        in_ch: int = 5,
        base_ch: int = 32,
        num_res: int = 2,
        max_disp: float = 60.0,
        out_act: str = "atan",
        corr_radius: int = 4,
        corr_base_ch: int = 24,
    ):
        super().__init__()
        self.max_disp = float(max_disp)
        self.out_act = str(out_act).lower().strip()
        self.corr_radius = int(corr_radius)

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, c1, 3, 1, 1),
            ResBlockGN(c1),
        )
        self.down1 = DownGN(c1, c2, num_res=num_res)
        self.down2 = DownGN(c2, c3, num_res=num_res)
        self.down3 = DownGN(c3, c4, num_res=num_res)
        self.down4 = DownGN(c4, c5, num_res=num_res)

        self.bottleneck = nn.Sequential(
            ResBlockGN(c5),
            ResBlockGN(c5),
        )

        self.up4 = UpGN(c5, c4, num_res=num_res)
        self.up3 = UpGN(c4, c3, num_res=num_res)
        self.up2 = UpGN(c3, c2, num_res=num_res)
        self.up1 = UpGN(c2, c1, num_res=num_res)

        self.corr_enc = CorrEncoder(in_ch=2, base_ch=corr_base_ch, num_res=1)

        cost_volume_channels = (2 * self.corr_radius + 1) ** 2
        self.fuse_corr = nn.Sequential(
            nn.Conv2d(c4 + 2 * cost_volume_channels, c4, 1, bias=False),
            nn.GroupNorm(_choose_gn_groups(c4), c4, eps=1e-5, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlockGN(c4),
            ResBlockGN(c4),
        )

        self.head_f01 = nn.Conv2d(c1, 2, 3, padding=1)
        self.head_f10 = nn.Conv2d(c1, 2, 3, padding=1)

        for head in (self.head_f01, self.head_f10):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _flow_act(self, raw: torch.Tensor) -> torch.Tensor:
        if self.out_act == "tanh":
            return torch.tanh(raw) * self.max_disp
        if self.out_act == "softsign":
            return (raw / (1.0 + raw.abs())) * self.max_disp
        return (2.0 / math.pi) * torch.atan(raw) * self.max_disp

    def forward(self, x: torch.Tensor, dcond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bidirectional dense flows ``(F01, F10)`` in pixel units."""
        del dcond  # Reserved for compatibility with older training code.

        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)

        i0_grad = torch.cat([x[:, 0:1, ...], x[:, 3:4, ...]], dim=1)
        i1_grad = torch.cat([x[:, 1:2, ...], x[:, 4:5, ...]], dim=1)
        f0 = self.corr_enc(i0_grad)
        f1 = self.corr_enc(i1_grad)
        cv01 = local_correlation(f0, f1, radius=self.corr_radius)
        cv10 = local_correlation(f1, f0, radius=self.corr_radius)

        s4_fused = self.fuse_corr(torch.cat([s4, cv01, cv10], dim=1))

        s5 = self.down4(s4_fused)
        bottleneck = self.bottleneck(s5)

        d4 = self.up4(bottleneck, s4_fused)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        f01 = self._flow_act(self.head_f01(d1))
        f10 = self._flow_act(self.head_f10(d1))
        return f01, f10
