"""
unet.py — Diffusion U-Net with SCADO attention at bottleneck and mid layers.

Architecture
────────────
Encoder:  DownBlock × L  (ResNet + optional standard self-attn)
Bottleneck: ResBlock + SCADO + ResBlock
Decoder:  UpBlock × L   (ResNet + optional standard self-attn)
Output:   Conv → noise prediction ε_θ

Conditioning signals injected via AdaGN (adaptive group norm):
  • timestep t    (sinusoidal embedding)
  • landmark embed (from LandmarkEncoder)
  • masked image + mask concatenated to input channels
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.scado_attention import SCADOAttention
from models.landmark_encoder import LandmarkEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal timestep embedding.  t: (B,) → (B, dim)"""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepMLP(nn.Module):
    """Projects sinusoidal embedding to conditioning vector."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, t_emb):
        return self.net(t_emb)


# ──────────────────────────────────────────────────────────────────────────────
# Residual Block with AdaGN conditioning
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    ResNet block with adaptive group normalisation.

    Conditioning: scale/shift derived from combined (time + landmark) embed.
    Optionally uses gradient checkpointing.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        # AdaGN: predict (scale, shift) from conditioning
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def _forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # AdaGN conditioning
        scale_shift = self.cond_proj(cond)               # B×(2*out_ch)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)

    def forward(self, x, cond):
        if self.use_checkpoint and self.training:
            def _checkpoint_fn(x_):
                return self._forward(x_, cond)
                
            return checkpoint(_checkpoint_fn, x)
        return self._forward(x, cond)


# ──────────────────────────────────────────────────────────────────────────────
# Standard Self-Attention (non-SCADO) for intermediate resolutions
# ──────────────────────────────────────────────────────────────────────────────

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(2, 0, 1)  # B×HW×C
        h, _ = self.attn(h, h, h)
        h = h.permute(1, 2, 0).view(B, C, H, W)
        return x + h


# ──────────────────────────────────────────────────────────────────────────────
# Down / Up blocks
# ──────────────────────────────────────────────────────────────────────────────

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, num_res=2, attn=False,
                 dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResBlock(in_ch if i == 0 else out_ch, out_ch, cond_dim,
                     dropout, use_checkpoint)
            for i in range(num_res)
        ])
        self.attns = nn.ModuleList([
            SelfAttentionBlock(out_ch) if attn else nn.Identity()
            for _ in range(num_res)
        ])
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, cond):
        skips = []
        for res, attn in zip(self.resnets, self.attns):
            x = res(x, cond)
            x = attn(x)
            skips.append(x)
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim, num_res=2, attn=False,
                 dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResBlock(in_ch + skip_ch if i == 0 else out_ch, out_ch, cond_dim,
                     dropout, use_checkpoint)
            for i in range(num_res)
        ])
        self.attns = nn.ModuleList([
            SelfAttentionBlock(out_ch) if attn else nn.Identity()
            for _ in range(num_res)
        ])
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)

    def forward(self, x, skips, cond):
        x = self.upsample(x)
        for i, (res, attn) in enumerate(zip(self.resnets, self.attns)):
            skip = skips.pop()
            if i == 0:
                x = torch.cat([x, skip], dim=1)
            x = res(x, cond)
            x = attn(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full Diffusion U-Net
# ──────────────────────────────────────────────────────────────────────────────

class DiffusionUNet(nn.Module):
    """
    U-Net for diffusion inpainting.

    Input channels: noisy_image(3) + masked_image(3) + mask(1) = 7
    Conditioning  : timestep + landmark embed → AdaGN
    SCADO         : applied at bottleneck (and optionally deepest mid layer)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg["model"]
        uc = cfg["unet"]
        sc = cfg["scado"]

        image_size     = mc["image_size"]
        in_ch          = 7          # noisy(3) + masked(3) + mask(1)
        out_ch         = mc["out_channels"]
        base_ch        = uc["base_channels"]
        ch_mult        = uc["channel_multipliers"]
        num_res        = uc["num_res_blocks"]
        attn_res       = set(uc["attention_resolutions"])
        dropout        = uc["dropout"]
        use_ckpt       = uc["use_gradient_checkpointing"]
        num_landmarks  = mc["num_landmarks"]
        embed_dim      = sc["embed_dim"]

        # ── conditioning embedding ─────────────────────────────────────
        time_dim = base_ch * 4
        self.time_dim = base_ch  
        self.time_mlp = TimestepMLP(base_ch, time_dim)

        self.lm_encoder = LandmarkEncoder(num_landmarks, embed_dim)
        self.lm_proj = nn.Linear(embed_dim, time_dim)

        cond_dim = time_dim  # time + landmark fused

        # ── input conv ────────────────────────────────────────────────
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ── encoder ───────────────────────────────────────────────────
        channels = [base_ch]
        ch = base_ch
        self.down_blocks = nn.ModuleList()
        cur_res = image_size

        for mult in ch_mult:
            out = base_ch * mult
            use_attn = cur_res in attn_res
            self.down_blocks.append(
                DownBlock(ch, out, cond_dim, num_res, use_attn, dropout, use_ckpt)
            )
            channels.append(out)
            ch = out
            cur_res //= 2

        # ── bottleneck ────────────────────────────────────────────────
        self.bot_res1 = ResBlock(ch, ch, cond_dim, dropout, use_ckpt)
        self.scado = SCADOAttention(
            embed_dim=ch,            # matches bottleneck channels
            num_heads=sc["num_heads"],
            num_landmarks=num_landmarks,
            landmark_sigma=sc["landmark_sigma"],
            graph_hidden_dim=sc["graph_hidden_dim"],
        )
        self.bot_res2 = ResBlock(ch, ch, cond_dim, dropout, use_ckpt)

        # ── decoder ───────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for mult in reversed(ch_mult):
            skip_ch = channels.pop()
            out = base_ch * mult
            use_attn = (cur_res * 2) in attn_res
            self.up_blocks.append(
                UpBlock(ch, skip_ch, out, cond_dim, num_res, use_attn, dropout, use_ckpt)
            )
            ch = out
            cur_res *= 2

        # ── output head ───────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_noisy: torch.Tensor,         # B×3×H×W  noisy image
        t: torch.Tensor,               # B         timestep indices
        masked_image: torch.Tensor,    # B×3×H×W
        mask: torch.Tensor,            # B×1×H×W
        landmarks: torch.Tensor,       # B×K×2
    ):
        # ── conditioning ──────────────────────────────────────────────
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)                   # B×cond_dim

        lm_emb = self.lm_encoder(landmarks)            # B×embed_dim
        lm_emb = self.lm_proj(lm_emb)                  # B×cond_dim

        cond = t_emb + lm_emb                           # B×cond_dim

        # ── input ─────────────────────────────────────────────────────
        x = torch.cat([x_noisy, masked_image, mask], dim=1)  # B×7×H×W
        x = self.in_conv(x)

        # ── encoder ───────────────────────────────────────────────────
        all_skips = []
        for block in self.down_blocks:
            x, skips = block(x, cond)
            all_skips.append(skips)

        # ── bottleneck with SCADO ──────────────────────────────────────
        x = self.bot_res1(x, cond)

        # Downsample mask to bottleneck resolution for SCADO
        bot_h, bot_w = x.shape[2], x.shape[3]
        mask_bot = F.interpolate(mask.float(), (bot_h, bot_w), mode="nearest")

        x, sym_loss = self.scado(x, mask_bot, landmarks)
        x = self.bot_res2(x, cond)

        # ── decoder ───────────────────────────────────────────────────
        for block, skips in zip(self.up_blocks, reversed(all_skips)):
            x = block(x, skips, cond)

        # ── output ────────────────────────────────────────────────────
        x = F.silu(self.out_norm(x))
        noise_pred = self.out_conv(x)

        return noise_pred, sym_loss
