"""
SDAF: Semantic–Detail Adaptive Frequency Fusion
===============================================

A neck core module deployed at the P3/8 top-down fusion node.
It replaces the standard Concat + C3k2 fusion block with an
adaptive frequency-domain fusion that uses:

- Cross-scale path competition (semantic vs lateral gating)
- External detail prior from RDPD (optional, for 3-input mode)
- Compressed-channel DCT with fixed soft radial band basis
- Sample-adaptive band weighting via MLP conditioned on
  feature statistics and spatial priors
- Bounded frequency residual with spatial guidance
  (applied AFTER IDCT, not in frequency domain)

Key constraints:
- Only ONE DCT path (compressed channels, not full c2)
- Fixed band basis (not learnable H×W matrix)
- Spatial prior only participates post-IDCT
- frequency_gamma initialised near zero for stable training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT

from ..modules.conv import Conv

__all__ = ["SDAF"]


class FixedRadialBandBasis:
    """Fixed soft radial frequency band basis.

    Constructs K soft bands over normalised radial frequency rho in [0, 1].
    The basis is NOT learnable, does NOT live in the optimizer, and is
    cached per (H, W, device, dtype) up to a maximum of 4 entries.

    Each band k is centred at mu_k with Gaussian falloff:
        B_k(u,v) = exp(-(rho - mu_k)^2 / (2 * sigma^2))
    then normalised so sum_k B_k = 1 at each (u, v).
    """

    _cache: dict = {}  # key -> tensor
    _max_cache_size: int = 4

    def __init__(self, num_bands: int = 4, sigma: float = 0.18):
        self.num_bands = num_bands
        self.sigma = sigma
        # Equal-spaced band centres
        self.mu = torch.linspace(0.0, 1.0, num_bands)  # e.g. [0, 1/3, 2/3, 1]

    def get_basis(self, H: int, W: int, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
        """Return band basis [K, H, W] for the given spatial size.

        The basis is cached to avoid recomputation. If the device or dtype
        of a cached entry no longer matches, it is rebuilt.
        """
        key = (H, W, str(device), str(dtype))
        if key in self._cache:
            cached = self._cache[key]
            if cached.device == device and cached.dtype == dtype:
                return cached

        # Build basis
        u = torch.linspace(0, 1, W, device=device, dtype=dtype)
        v = torch.linspace(0, 1, H, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # [H, W], [H, W]
        rho = torch.sqrt(uu ** 2 + vv ** 2) / math.sqrt(2.0)  # [H, W], range ~[0,1]

        mu = self.mu.to(device=device, dtype=dtype)
        sigma2 = self.sigma ** 2
        # [K, H, W]
        B = torch.exp(-((rho.unsqueeze(0) - mu.view(-1, 1, 1)) ** 2) / (2.0 * sigma2))
        # Normalise per spatial location so sum_k B_k = 1
        B = B / (B.sum(dim=0, keepdim=True) + 1e-8)

        # Evict oldest if cache full
        if len(self._cache) >= self._max_cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = B.detach()  # permanently detached
        return B


class SDAF(nn.Module):
    """Semantic–Detail Adaptive Frequency Fusion.

    Deployed at the P3/8 top-down fusion node in the FPN neck.
    Replaces the standard [Concat + C3k2] fusion block.

    Args:
        inc: List of input channel counts (2 or 3 elements).
             [high_semantic, lateral_detail]  or
             [high_semantic, lateral_detail, rdpd_packed_ch]
        c2: Output channels (should match the replaced C3k2 output).
        reduction: Compression ratio for DCT channel bottleneck (default 4).
        num_bands: Number of fixed radial frequency bands (default 4).
        band_sigma: Gaussian sigma for band basis (default 0.18).

    Input:
        x: list of tensors with 2 or 3 elements:
            - high_semantic: upsampled P4 feature  [B, C_hi, H, W]
            - lateral_detail: backbone P3 feature   [B, C_lo, H, W]
            - rdpd_packed (optional): RDPD output   [B, C_packed, H/2, W/2]
              (last channel is the external detail prior)

    Output:
        Y: fused tensor [B, c2, H, W]
    """

    def __init__(
        self,
        inc: list,
        c2: int,
        reduction: int = 4,
        num_bands: int = 4,
        use_spatial_guidance: bool = True,
        learnable_band_weights: bool = True,
        use_soft_bands: bool = True,
        band_sigma: float = 0.18,
    ):
        super().__init__()
        self.c2 = c2
        self.num_inputs = len(inc)
        self.num_bands = num_bands
        self.use_spatial_guidance = use_spatial_guidance
        self.learnable_band_weights = learnable_band_weights
        self.use_soft_bands = use_soft_bands

        # ── 8.1 Channel alignment ──
        self.align_high = Conv(inc[0], c2, k=1)
        self.align_lateral = Conv(inc[1], c2, k=1)

        # ── 8.2 Cross-scale path competition gate ──
        gate_in_2c = c2 * 2
        self.path_dw = nn.Conv2d(gate_in_2c, gate_in_2c, kernel_size=3,
                                 padding=1, groups=gate_in_2c, bias=False)
        self.path_bn = nn.BatchNorm2d(gate_in_2c)
        self.path_act = nn.SiLU()
        self.path_pw = nn.Conv2d(gate_in_2c, c2 * 2, kernel_size=1, bias=False)

        # ── 8.3 External prior fusion (when rdpd_packed is provided) ──
        self.has_external = (self.num_inputs >= 3)
        self._external_prior_used: bool = False  # internal flag, updated in forward
        if self.has_external:
            self.prior_fuse_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

        # ── 8.4 Compressed-channel frequency bottleneck ──
        Cr = max(16, c2 // reduction)
        self.Cr = Cr
        self.compress = nn.Sequential(
            nn.Conv2d(c2, Cr, kernel_size=1, bias=False),
            nn.BatchNorm2d(Cr),
            nn.SiLU(),
        )

        # ── 8.5 Fixed radial band basis ──
        self.band_basis = FixedRadialBandBasis(num_bands=num_bands, sigma=band_sigma)

        # ── 8.6 Sample-adaptive band weight MLP ──
        # Input: [zg(Cr), s_mean(1), s_std(1)] = Cr + 2
        mlp_in = Cr + 2
        mlp_hidden = max(8, Cr // 2)
        self.band_mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, num_bands),
        )

        # ── 8.8 Spatial guidance gate (post-IDCT) ──
        # Input: [Hf_norm(1), S(1)] = 2 channels
        self.spatial_dw = nn.Conv2d(2, 2, kernel_size=3, padding=1,
                                    groups=2, bias=False)
        self.spatial_act = nn.SiLU()
        self.spatial_pw = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        # ── 8.9 Bounded frequency residual ──
        self.expand = nn.Conv2d(Cr, c2, kernel_size=1, bias=False)
        # Initialize so sigmoid ≈ 0.01
        self.frequency_gamma_raw = nn.Parameter(torch.tensor(-4.5951198))

        # ── 8.10 Debug state (detached, not in state_dict) ──
        self.register_buffer("last_band_weights", torch.zeros(1, num_bands), persistent=False)
        self.register_buffer("last_prior", torch.zeros(1, 1, 1, 1), persistent=False)
        self.register_buffer("last_spatial_mask", torch.zeros(1, 1, 1, 1), persistent=False)
        self.register_buffer("last_used_external_prior", torch.tensor(False), persistent=False)

    def forward(self, x: list) -> torch.Tensor:
        assert len(x) in (2, 3), f"SDAF expects 2 or 3 inputs, got {len(x)}"

        high_semantic: torch.Tensor = x[0]
        lateral_detail: torch.Tensor = x[1]

        _, _, H_hi, W_hi = high_semantic.shape
        _, _, H, W = lateral_detail.shape  # target spatial size

        # ── 8.1 Size & channel alignment ──
        if H_hi != H or W_hi != W:
            high_semantic = F.interpolate(
                high_semantic, size=(H, W), mode="nearest"
            )
        Xh: torch.Tensor = self.align_high(high_semantic)  # [B, c2, H, W]
        Xl: torch.Tensor = self.align_lateral(lateral_detail)  # [B, c2, H, W]

        # ── 8.2 Cross-scale path competition ──
        cat_hl: torch.Tensor = torch.cat([Xh, Xl], dim=1)  # [B, 2c2, H, W]
        path_feat: torch.Tensor = self.path_dw(cat_hl)
        path_feat = self.path_bn(path_feat)
        path_feat = self.path_act(path_feat)
        path_logits: torch.Tensor = self.path_pw(path_feat)  # [B, 2c2, H, W]

        B, _, H_out, W_out = path_logits.shape
        path_logits = path_logits.view(B, 2, self.c2, H_out, W_out)
        path_weights: torch.Tensor = F.softmax(path_logits, dim=1)  # [B, 2, c2, H, W]

        Wh: torch.Tensor = path_weights[:, 0, :, :, :]  # [B, c2, H, W]
        Wl: torch.Tensor = path_weights[:, 1, :, :, :]  # [B, c2, H, W]

        # Base fusion
        X0: torch.Tensor = Wh * Xh + Wl * Xl  # [B, c2, H, W]

        # Local detail prior
        Sl: torch.Tensor = Wl.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # ── 8.3 External prior fusion ──
        if len(x) >= 3:
            self._external_prior_used = True
            rdpd_packed: torch.Tensor = x[2]
            external_prior: torch.Tensor = rdpd_packed[:, -1:, :, :]  # [B, 1, Hrp, Wrp]

            # Bilinear interpolate to target spatial size
            if external_prior.shape[-2:] != (H, W):
                external_prior = F.interpolate(
                    external_prior, size=(H, W),
                    mode="bilinear", align_corners=False
                )

            # Fuse Sl and external_prior → S
            prior_cat: torch.Tensor = torch.cat([Sl, external_prior], dim=1)  # [B, 2, H, W]
            S: torch.Tensor = torch.sigmoid(self.prior_fuse_conv(prior_cat))  # [B, 1, H, W]
        else:
            self._external_prior_used = False
            S: torch.Tensor = Sl

        # ── 8.4 Compressed-channel frequency bottleneck ──
        Z: torch.Tensor = self.compress(X0)  # [B, Cr, H, W]

        # ── 8.5 Fixed band basis ──
        Bk: torch.Tensor = self.band_basis.get_basis(H, W, Z.device, Z.dtype)  # [K, H, W]
        # Bk is fixed, not in optimizer

        if not getattr(self, 'use_soft_bands', True):
            # HS-FPN style hard threshold: keep only highest-frequency band
            # (band K-1, centred near rho=1) as a binary mask
            Bk = (Bk[-1:] > 0.5).float()  # [1, H, W] binary mask
            a_hard = torch.ones(B, 1, device=Z.device, dtype=Z.dtype)
            Gf: torch.Tensor = a_hard.unsqueeze(-1).unsqueeze(-1) * Bk  # [B, 1, H, W]
            # skip MLP path below, jump directly to DCT
        else:
            # ── 8.6 Sample-adaptive band weights ──
            if getattr(self, 'learnable_band_weights', True):
                zg: torch.Tensor = F.adaptive_avg_pool2d(Z, 1).view(B, -1)  # [B, Cr]
                s_mean: torch.Tensor = S.mean(dim=[2, 3])  # [B, 1]
                s_std: torch.Tensor = S.std(dim=[2, 3], unbiased=False)  # [B, 1]
                q: torch.Tensor = torch.cat([zg, s_mean, s_std], dim=1)  # [B, Cr+2]

                a_logits: torch.Tensor = self.band_mlp(q)  # [B, K]
                a: torch.Tensor = F.softmax(a_logits, dim=1)  # [B, K]
            else:
                # Fixed uniform weights — ablates sample-adaptive claim
                a = torch.ones(B, self.num_bands, device=Z.device, dtype=Z.dtype)
                a = a / self.num_bands  # [B, K] uniform

            # Dynamic frequency gate
            Gf: torch.Tensor = torch.einsum("bk,khw->bhw", a, Bk)  # [B, H, W]
            Gf = Gf.unsqueeze(1)  # [B, 1, H, W]

        # ── 8.7 DCT / IDCT with AMP safety ──
        original_dtype = Z.dtype
        Z32: torch.Tensor = Z.float()
        Gf32: torch.Tensor = Gf.float()

        F32: torch.Tensor = DCT.dct_2d(Z32, norm="ortho")
        F_filtered: torch.Tensor = F32 * Gf32
        Zf32: torch.Tensor = DCT.idct_2d(F_filtered, norm="ortho")

        Zf: torch.Tensor = Zf32.to(original_dtype)

        # ── 8.8 Spatial guidance (post-IDCT) ──
        if getattr(self, 'use_spatial_guidance', True):
            Hf: torch.Tensor = Zf.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
            # Sample-wise normalisation
            Hf_norm: torch.Tensor = Hf / (Hf.mean(dim=[2, 3], keepdim=True) + 1e-6)

            spatial_in: torch.Tensor = torch.cat([Hf_norm, S], dim=1)  # [B, 2, H, W]
            spatial_feat: torch.Tensor = self.spatial_dw(spatial_in)
            spatial_feat = self.spatial_act(spatial_feat)
            M: torch.Tensor = torch.sigmoid(self.spatial_pw(spatial_feat))  # [B, 1, H, W]
        else:
            # No spatial guidance — ablates decoupled design claim
            M = torch.ones(B, 1, H, W, device=Zf.device, dtype=Zf.dtype)

        # ── 8.9 Bounded frequency residual ──
        Rf: torch.Tensor = self.expand(Zf)  # [B, c2, H, W]
        frequency_gamma: torch.Tensor = torch.sigmoid(self.frequency_gamma_raw)
        Y: torch.Tensor = X0 + frequency_gamma * M * Rf  # [B, c2, H, W]

        # ── 8.10 Debug state ──
        self.last_band_weights = a.detach()
        self.last_prior = S.detach()
        self.last_spatial_mask = M.detach()
        self.last_used_external_prior = torch.tensor(
            self._external_prior_used, device=Y.device
        )

        return Y
