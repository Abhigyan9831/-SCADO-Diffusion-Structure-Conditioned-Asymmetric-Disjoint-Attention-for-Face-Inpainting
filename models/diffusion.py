"""
diffusion.py — DDPM forward process + DDIM reverse sampling.

Forward (training):
    q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1 - ᾱ_t) I)
    Loss = ||ε - ε_θ(x_t, t, c)||^2

Reverse (inference, DDIM):
    x_{t-1} = sqrt(ᾱ_{t-1}) · x̂_0
             + sqrt(1 - ᾱ_{t-1} - σ_t²) · ε_θ
             + σ_t · ε        (σ_t = 0 for deterministic DDIM)
"""

import math
import numpy as np
import torch
import torch.nn as nn


def _cosine_beta_schedule(T: int):
    """Improved cosine noise schedule (Nichol & Dhariwal 2021)."""
    s = 0.008
    steps = np.arange(T + 1)
    alphas_cumprod = np.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 0.0001, 0.9999)


def _linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)


class GaussianDiffusion(nn.Module):
    """
    Wraps the U-Net into a complete diffusion model.

    Args:
        model           : DiffusionUNet
        num_timesteps   : T (default 1000)
        noise_schedule  : 'cosine' | 'linear'
        sampling_steps  : DDIM steps at inference (default 50)
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        noise_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        sampling_steps: int = 50,
    ):
        super().__init__()
        self.model = model
        self.T = num_timesteps
        self.sampling_steps = sampling_steps

        # Build schedule
        if noise_schedule == "cosine":
            betas = _cosine_beta_schedule(num_timesteps)
        else:
            betas = _linear_beta_schedule(num_timesteps, beta_start, beta_end)

        betas = torch.from_numpy(betas).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register all schedule tensors as buffers (moved with .to(device))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer("log_one_minus_alphas_cumprod", (1 - alphas_cumprod).log())
        self.register_buffer("sqrt_recip_alphas_cumprod", (1 / alphas_cumprod).sqrt())
        self.register_buffer(
            "sqrt_recip_alphas_cumprod_minus_one", (1 / alphas_cumprod - 1).sqrt()
        )

        # Posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_variance.clamp(min=1e-20).log()
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def _extract(self, buf: torch.Tensor, t: torch.Tensor, shape):
        """Gather schedule values at timestep t and broadcast to shape."""
        vals = buf.gather(0, t)
        while len(vals.shape) < len(shape):
            vals = vals.unsqueeze(-1)
        return vals.expand(shape)

    # ──────────────────────────────────────────────────────────────────
    # Forward diffusion  q(x_t | x_0)
    # ──────────────────────────────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        """
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_acp * x0 + sqrt_om * noise, noise

    # ──────────────────────────────────────────────────────────────────
    # Training loss
    # ──────────────────────────────────────────────────────────────────
    def compute_loss(
        self,
        x0: torch.Tensor,          # B×3×H×W  ground truth
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        landmarks: torch.Tensor,
    ):
        """
        Returns dict with 'diffusion_loss' (MSE on noise) and 'sym_loss'.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)

        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise)

        noise_pred, sym_loss = self.model(x_t, t, masked_image, mask, landmarks)

        diffusion_loss = ((noise_pred - noise) ** 2).mean()
        x0_pred = self.predict_x0(x_t, t, noise_pred)
        return {"diffusion_loss": diffusion_loss, "sym_loss": sym_loss, "pred_img": x0_pred.clamp(-1, 1) }

    # ──────────────────────────────────────────────────────────────────
    # Predict x_0 from x_t and predicted noise
    # ──────────────────────────────────────────────────────────────────
    def predict_x0(self, x_t, t, noise_pred):
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_rm1 = self._extract(self.sqrt_recip_alphas_cumprod_minus_one, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_rm1 * noise_pred

    # ──────────────────────────────────────────────────────────────────
    # DDIM sampling
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def ddim_sample(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        landmarks: torch.Tensor,
        eta: float = 0.0,          # 0 = deterministic DDIM
    ):
        """
        Full reverse diffusion from x_T to x_0 using DDIM.

        Returns final predicted image and list of (t, attn_map) pairs
        from SCADO for visualisation.
        """
        B, C, H, W = masked_image.shape
        device = masked_image.device

        # DDIM timestep subsequence
        ddim_steps = self.sampling_steps
        step_ratio = self.T // ddim_steps
        timesteps = list(reversed(range(0, self.T, step_ratio)))[:ddim_steps]

        x = torch.randn(B, C, H, W, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            noise_pred, _ = self.model(x, t, masked_image, mask, landmarks)
            x0_pred = self.predict_x0(x, t, noise_pred)
            x0_pred = x0_pred.clamp(-1, 1)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t = self.alphas_cumprod[t_val]
                alpha_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_t = self.alphas_cumprod[t_val]
                alpha_prev = torch.ones(1, device=device)

            sigma_t = (
                eta
                * ((1 - alpha_prev) / (1 - alpha_t)).sqrt()
                * (1 - alpha_t / alpha_prev).sqrt()
            )
            dir_xt = (1 - alpha_prev - sigma_t ** 2).sqrt() * noise_pred
            rand = sigma_t * torch.randn_like(x) if eta > 0 else 0.0

            x = alpha_prev.sqrt() * x0_pred + dir_xt + rand

        # Composite: keep visible region from input
        out = x * mask + masked_image * (1 - mask)
        return out.clamp(-1, 1)
