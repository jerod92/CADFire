"""
Spectral information pathway for the CADFire model.

Runs in parallel with the VisionEncoder (UNet-Down pathway).  Uses custom
SpectralConv2d layers that operate in the Fourier domain to capture global
frequency-domain structure (edges, periodic patterns, spatial frequency
content) that spatial convolutions tend to miss at early layers.

Architecture
────────────
Input: (B, C_in, H, W)

  Stem               : 1×1 Conv  → (B, C, H, W)                C = base_channels

  Stage-1 (H×W)      : SpectralConv2d(C → C) || identity branch
                       cat → (B, 2C, H, W)
                       3×3 Conv stride-2 → (B, 2C, H/2, W/2)

  Stage-2 (H/2×W/2)  : SpectralConv2d(2C → 2C) || identity branch
                       cat → (B, 4C, H/2, W/2)
                       3×3 Conv stride-2 → (B, 4C, H/4, W/4)

  Stage-3 (H/4×W/4)  : SpectralConv2d(4C → 4C) || identity branch
                       cat → (B, 8C, H/4, W/4)
                       3×3 Conv stride-2 → (B, 8C, H/8, W/8)

  Global Pool + FC   → (B, fusion_dim)

The SpectralConv2d is inspired by the Fourier Neural Operator (FNO) approach:
it applies a learned linear mixing in the frequency domain (lowest K modes),
which is equivalent to a globally-receptive convolution in the spatial domain.

References
──────────
Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential
Equations". https://arxiv.org/abs/2010.08895
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Custom spectral convolution layer ─────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """
    Learnable 2-D spectral convolution via real FFT.

    For each (B, C_in, H, W) input:
      1. Compute 2-D real FFT → (B, C_in, H, W//2+1) complex spectrum.
      2. Keep the lowest ``modes_h`` × ``modes_w`` Fourier modes.
      3. Apply a learned complex linear mixing across channels for those modes.
      4. Inverse FFT back to spatial domain → (B, C_out, H, W).

    The mixing is parameterised by separate real and imaginary weight tensors
    (more numerically stable than native complex parameters).
    """

    def __init__(self, in_ch: int, out_ch: int,
                 modes_h: int = 16, modes_w: int = 16):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_ch * out_ch) ** 0.5
        # Real and imaginary parts of the spectral weight tensor
        self.weight_r = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_h, modes_w)
        )
        self.weight_i = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_h, modes_w)
        )

    def _complex_mul(self,
                     x_r: torch.Tensor, x_i: torch.Tensor,
                     w_r: torch.Tensor, w_i: torch.Tensor,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complex matrix multiply: (x_r + j·x_i) × (w_r + j·w_i).

        All tensors have shape (B, C_in, mh, mw) / (C_in, C_out, mh, mw).
        """
        out_r = (torch.einsum('bimn,iomn->bomn', x_r, w_r)
                 - torch.einsum('bimn,iomn->bomn', x_i, w_i))
        out_i = (torch.einsum('bimn,iomn->bomn', x_r, w_i)
                 + torch.einsum('bimn,iomn->bomn', x_i, w_r))
        return out_r, out_i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            out: (B, C_out, H, W)  — spectral-domain processed features
        """
        B, C, H, W = x.shape
        W_half = W // 2 + 1

        # 2-D real FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W_half) cfloat

        # Clamp modes to available spectrum
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W_half)

        # Allocate zero output spectrum
        out_ft_r = torch.zeros(B, self.out_ch, H, W_half, device=x.device)
        out_ft_i = torch.zeros(B, self.out_ch, H, W_half, device=x.device)

        # Mix the kept modes
        wr = self.weight_r[:, :, :mh, :mw]
        wi = self.weight_i[:, :, :mh, :mw]
        x_sub_r = x_ft[:, :, :mh, :mw].real
        x_sub_i = x_ft[:, :, :mh, :mw].imag

        mixed_r, mixed_i = self._complex_mul(x_sub_r, x_sub_i, wr, wi)
        out_ft_r[:, :, :mh, :mw] = mixed_r
        out_ft_i[:, :, :mh, :mw] = mixed_i

        # Recombine into complex and IFFT back to spatial
        out_ft = torch.complex(out_ft_r, out_ft_i)
        out    = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')  # (B, C_out, H, W)
        return out


# ── Spectral Encoder ──────────────────────────────────────────────────────────

class SpectralEncoder(nn.Module):
    """
    Spectral information pathway: parallel branch to the VisionEncoder.

    At each stage the input is passed through a SpectralConv2d (global
    frequency mixing) and the result is *concatenated* with the original
    features before a strided spatial convolution that downsamples.  This
    lets the network blend spatial and spectral information at every scale.

    Output is a single global feature vector (B, fusion_dim) suitable for
    concatenation inside the FusionBridge alongside the spatial vision
    encoding, text encoding, and state vector.
    """

    def __init__(self, in_channels: int, base_channels: int = 32,
                 fusion_dim: int = 256,
                 modes_h: int = 16, modes_w: int = 16):
        super().__init__()
        C = base_channels

        # 1×1 stem – channel reduction without touching spatial structure
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        # ── Stage 1: full resolution ───────────────────────────────────────
        # SpectralConv: (B, C, H, W) → (B, C, H, W)
        # concat with identity → (B, 2C, H, W)
        # mix + stride-2 → (B, 2C, H/2, W/2)
        self.spec1 = SpectralConv2d(C, C, modes_h, modes_w)
        self.mix1 = nn.Sequential(
            nn.Conv2d(C * 2, C * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 2),
            nn.GELU(),
        )

        # ── Stage 2: H/2 × W/2 ────────────────────────────────────────────
        self.spec2 = SpectralConv2d(C * 2, C * 2, modes_h, modes_w)
        self.mix2 = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 4),
            nn.GELU(),
        )

        # ── Stage 3: H/4 × W/4 ────────────────────────────────────────────
        self.spec3 = SpectralConv2d(C * 4, C * 4, modes_h, modes_w)
        self.mix3 = nn.Sequential(
            nn.Conv2d(C * 8, C * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 8),
            nn.GELU(),
        )

        # Global feature head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc   = nn.Linear(C * 8, fusion_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)  — same image tensor fed to VisionEncoder
        Returns:
            spectral_feat: (B, fusion_dim)
        """
        # Stem
        x = self.stem(x)                                # (B, C, H, W)

        # Stage 1
        s1 = self.spec1(x)                              # (B, C, H, W)
        x  = self.mix1(torch.cat([x, s1], dim=1))       # (B, 2C, H/2, W/2)

        # Stage 2
        s2 = self.spec2(x)                              # (B, 2C, H/2, W/2)
        x  = self.mix2(torch.cat([x, s2], dim=1))       # (B, 4C, H/4, W/4)

        # Stage 3
        s3 = self.spec3(x)                              # (B, 4C, H/4, W/4)
        x  = self.mix3(torch.cat([x, s3], dim=1))       # (B, 8C, H/8, W/8)

        # Global pool → feature vector
        g = self.global_pool(x).flatten(1)              # (B, 8C)
        return self.global_fc(g)                        # (B, fusion_dim)
