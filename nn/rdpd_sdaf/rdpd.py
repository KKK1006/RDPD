"""
RDPD: Residual Detail-Preserving Downsampling
=============================================

A backbone auxiliary structure that replaces a standard stride-2 Conv
at the P2/4 → P3/8 transition. It decomposes the downsampling into
three branches (semantic, learnable, smooth) and explicitly models the
residual detail that cannot be explained by smooth (low-frequency)
downsampling.

Key design:
- Semantic branch: standard Conv 3×3 stride=2 (preserves semantics)
- Learnable branch: depthwise+pointwise with residual detail extraction
- Smooth branch: AvgPool + pointwise (captures only low frequency)
- Path competition: softmax gating over [semantic, detail] paths
- Detail prior: channel-mean of detail gate weights, concatenated to output

Output: packed tensor [B, c2+1, H/2, W/2]
  - packed[:, :c2]  = main downsampled feature
  - packed[:, c2:]  = single-channel detail demand prior (no detach)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv

__all__ = ["RDPD"]


class RDPD(nn.Module):
    """Residual Detail-Preserving Downsampling.

    Replaces a standard stride-2 Conv in the backbone to preserve
    fine-grained spatial detail that would otherwise be lost during
    aggressive downsampling — critical for small object detection.

    Args:
        c1: Input channels.
        c2: Output feature channels (the detail prior adds 1 extra channel).

    Input:
        x: [B, c1, H, W]

    Output:
        packed: [B, c2 + 1, H/2, W/2]
    """

    def __init__(self, c1: int, c2: int, detach_sd: bool = False):
        super().__init__()
        self.detach_sd = detach_sd
        self.feature_channels = c2  # for inspection only

        # ── 7.1 Semantic branch ──
        self.semantic = Conv(c1, c2, k=3, s=2)

        # ── 7.2 Learnable downsampling response branch ──
        self.dw_conv = nn.Conv2d(c1, c1, kernel_size=3, stride=2,
                                 padding=1, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        self.dw_act = nn.SiLU()
        self.pw_conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)

        # ── 7.3 Smooth low-frequency branch ──
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.smooth_pw = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.smooth_bn = nn.BatchNorm2d(c2)

        # ── 7.4 Explicit detail residual ──
        self.detail_bn = nn.BatchNorm2d(c2)

        # ── 7.5 Path competition gate ──
        gate_in = c2 * 2  # concat [semantic, detail]
        self.gate_dw = nn.Conv2d(gate_in, gate_in, kernel_size=3,
                                 padding=1, groups=gate_in, bias=False)
        self.gate_bn = nn.BatchNorm2d(gate_in)
        self.gate_act = nn.SiLU()
        self.gate_pw = nn.Conv2d(gate_in, c2 * 2, kernel_size=1, bias=False)

        # ── 7.6 Stable residual output ──
        # Initialize so sigmoid ≈ 0.10
        self.detail_gamma_raw = nn.Parameter(torch.tensor(-2.1972246))

        # ── Output dimension sanity ──
        self._output_channels = c2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 7.1 Semantic branch ──
        Xs: torch.Tensor = self.semantic(x)  # [B, c2, H/2, W/2]

        # ── 7.2 Learnable downsampling response branch ──
        x_dw: torch.Tensor = self.dw_conv(x)       # [B, c1, H/2, W/2]
        x_dw = self.dw_bn(x_dw)
        x_dw = self.dw_act(x_dw)
        Xa: torch.Tensor = self.pw_bn(self.pw_conv(x_dw))  # [B, c2, H/2, W/2]
        # No activation after pointwise yet (design §7.2)

        # ── 7.3 Smooth low-frequency branch ──
        x_pool: torch.Tensor = self.avg_pool(x)     # [B, c1, H/2, W/2]
        Xl: torch.Tensor = self.smooth_bn(self.smooth_pw(x_pool))  # [B, c2, H/2, W/2]

        # ── 7.4 Explicit detail residual ──
        Xd_raw: torch.Tensor = Xa - Xl              # residual detail
        Xd: torch.Tensor = F.silu(self.detail_bn(Xd_raw))

        # ── 7.5 Path competition gate ──
        # Concatenate semantic and detail paths
        cat_sd: torch.Tensor = torch.cat([Xs, Xd], dim=1)  # [B, 2c2, H/2, W/2]
        gate_feat: torch.Tensor = self.gate_dw(cat_sd)
        gate_feat = self.gate_bn(gate_feat)
        gate_feat = self.gate_act(gate_feat)
        gate_logits: torch.Tensor = self.gate_pw(gate_feat)  # [B, 2c2, H/2, W/2]

        # Reshape to [B, 2, c2, H/2, W/2] and apply softmax over path dim
        B, _, H_out, W_out = gate_logits.shape
        gate_logits = gate_logits.view(B, 2, -1, H_out, W_out)  # [B, 2, c2, H/2, W/2]
        gate_weights: torch.Tensor = F.softmax(gate_logits, dim=1)  # softmax over path dim

        Ws: torch.Tensor = gate_weights[:, 0:1, :, :, :]  # [B, 1, c2, H, W] semantic weight
        Wd: torch.Tensor = gate_weights[:, 1:2, :, :, :]  # [B, 1, c2, H, W] detail weight

        # Squeeze the singleton path dimension
        Ws = Ws.squeeze(1)  # [B, c2, H/2, W/2]
        Wd = Wd.squeeze(1)  # [B, c2, H/2, W/2]

        # ── 7.5 Fusion ──
        mix: torch.Tensor = Ws * Xs + Wd * Xd  # [B, c2, H/2, W/2]

        # ── 7.6 Stable residual output ──
        detail_gamma: torch.Tensor = torch.sigmoid(self.detail_gamma_raw)
        Y: torch.Tensor = Xs + detail_gamma * (mix - Xs)  # [B, c2, H/2, W/2]

        # ── 7.7 Detail demand prior ──
        Sd: torch.Tensor = Wd.mean(dim=1, keepdim=True)  # [B, 1, H/2, W/2]
        if getattr(self, 'detach_sd', False):
            Sd = Sd.detach()  # cut gradient — ablates the closed-loop claim

        # ── Final packed output ──
        packed: torch.Tensor = torch.cat([Y, Sd], dim=1)  # [B, c2+1, H/2, W/2]

        return packed

    @property
    def output_channels(self) -> int:
        """Total output channels = feature_channels + 1 prior channel."""
        return self._output_channels
