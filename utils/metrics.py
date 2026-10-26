"""
metrics.py — Evaluation metrics: PSNR, SSIM, FID.

Usage:
    tracker = MetricTracker()
    tracker.update(pred, target)    # tensors in [-1,1]
    results = tracker.compute()
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> float:
    """
    Peak signal-to-noise ratio.
    Inputs in [-1,1] → range = 2.0.
    """
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(max_val / math.sqrt(mse))


# ──────────────────────────────────────────────────────────────────────────────
# SSIM
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(size=11, sigma=1.5, channels=3):
    x = torch.arange(size).float() - size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.outer(gauss)
    return kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size=11) -> float:
    """
    SSIM over a batch.  Inputs in [-1,1].
    Returns mean SSIM as float.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2  # constants for [-1,1] range approximation

    # Rescale to [0,1]
    p = (pred + 1) / 2
    t = (target + 1) / 2

    B, C, H, W = p.shape
    kernel = _gaussian_kernel(window_size, 1.5, C).to(p.device)

    def conv(x):
        return F.conv2d(x, kernel, padding=window_size // 2, groups=C)

    mu1, mu2 = conv(p), conv(t)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = conv(p * p) - mu1_sq
    sigma2_sq = conv(t * t) - mu2_sq
    sigma12   = conv(p * t) - mu12

    ssim_map = (
        (2 * mu12 + C1) * (2 * sigma12 + C2)
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )
    return ssim_map.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# FID helper (feature accumulator)
# ──────────────────────────────────────────────────────────────────────────────

class FIDFeatureAccumulator:
    """
    Accumulates InceptionV3 features for FID computation.
    Requires torchmetrics or clean-fid; falls back to a stub.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self._real_feats = []
        self._fake_feats = []
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            import torchvision.models as tvm
            inception = tvm.inception_v3(
                pretrained=True,
                transform_input=False,
            )
            inception.fc = torch.nn.Identity()
            inception.eval()
            self._model = inception.to(self.device)
        except Exception:
            self._model = None

    @torch.no_grad()
    def update(self, real: torch.Tensor, fake: torch.Tensor):
        if self._model is None:
            return
        real_r = F.interpolate(real, (299, 299), mode="bilinear", align_corners=False)
        fake_r = F.interpolate(fake, (299, 299), mode="bilinear", align_corners=False)
        # Convert [-1,1] to [0,1]
        real_r = (real_r + 1) / 2
        fake_r = (fake_r + 1) / 2
        self._real_feats.append(self._model(real_r.to(self.device)).cpu().numpy())
        self._fake_feats.append(self._model(fake_r.to(self.device)).cpu().numpy())

    def compute(self) -> float:
        if not self._real_feats:
            return -1.0
        try:
            from scipy import linalg
            real = np.concatenate(self._real_feats, axis=0)
            fake = np.concatenate(self._fake_feats, axis=0)
            mu_r, sigma_r = real.mean(0), np.cov(real, rowvar=False)
            mu_f, sigma_f = fake.mean(0), np.cov(fake, rowvar=False)
            diff = mu_r - mu_f
            cov_mean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
            if np.iscomplexobj(cov_mean):
                cov_mean = cov_mean.real
            fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * cov_mean))
            return fid
        except ImportError:
            return -1.0

    def reset(self):
        self._real_feats.clear()
        self._fake_feats.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Unified tracker
# ──────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    def __init__(self, device="cpu", compute_fid=True):
        self.psnr_vals = []
        self.ssim_vals = []
        self.fid_acc = FIDFeatureAccumulator(device) if compute_fid else None
        try:
            import lpips
            # Load LPIPS. Use verbose=False to reduce output. 
            # We initialize and then move to device to be safer on Windows.
            self.lpips_fn = lpips.LPIPS(net="alex", verbose=False)
            self.lpips_fn.to(device)
            self.lpips_fn.eval()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not initialize LPIPS: {e}. Skipping LPIPS metrics.")
            self.lpips_fn = None
        self.lpips_vals = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: B×3×H×W in [-1,1]"""
        pred_c = pred.clamp(-1, 1)
        target_c = target.clamp(-1, 1)

        for p, t in zip(pred_c, target_c):
            self.psnr_vals.append(psnr(p.unsqueeze(0), t.unsqueeze(0)))
            self.ssim_vals.append(ssim(p.unsqueeze(0), t.unsqueeze(0)))

        if self.fid_acc is not None:
            self.fid_acc.update(target_c, pred_c)

        if self.lpips_fn is not None:
            with torch.no_grad():
                lp = self.lpips_fn(
                    pred_c.to(next(self.lpips_fn.parameters()).device),
                    target_c.to(next(self.lpips_fn.parameters()).device)
                ).mean().item()
                self.lpips_vals.append(lp)

    def compute(self) -> dict:
        result = {
            "psnr": float(np.mean(self.psnr_vals)) if self.psnr_vals else 0.0,
            "ssim": float(np.mean(self.ssim_vals)) if self.ssim_vals else 0.0,
            "fid": self.fid_acc.compute() if self.fid_acc else -1.0,
            "lpips": float(np.mean(self.lpips_vals)) if self.lpips_vals else -1.0,
        }
        return result

    def reset(self):
        self.psnr_vals.clear()
        self.ssim_vals.clear()
        self.lpips_vals.clear()
        if self.fid_acc:
            self.fid_acc.reset()
        #if self.lpips_fn:          
       # pass
