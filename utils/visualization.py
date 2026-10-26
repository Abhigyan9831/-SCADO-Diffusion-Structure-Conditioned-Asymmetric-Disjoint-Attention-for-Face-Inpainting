"""
visualization.py — Save grids of input / masked / output / attention maps.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pathlib import Path


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert C×H×W tensor in [-1,1] to H×W×3 uint8 numpy array."""
    img = tensor.detach().cpu().float()
    img = (img.clamp(-1, 1) + 1) / 2         # [0,1]
    img = (img * 255).byte()
    return img.permute(1, 2, 0).numpy()


def save_sample_grid(
    images: dict,
    save_path: str,
    nrow: int = 4,
):
    """
    Save a grid comparing:  original | masked | predicted

    Args:
        images   : dict with keys 'original', 'masked', 'predicted' (B×3×H×W, [-1,1])
        save_path: path to write PNG
    """
    from PIL import Image as PILImage

    rows = []
    for key in ["original", "masked", "predicted"]:
        if key in images:
            rows.append(images[key].cpu())

    if not rows:
        return

    all_imgs = torch.cat(rows, dim=0)              # (3B)×3×H×W
    grid = vutils.make_grid(
        all_imgs.clamp(-1, 1), nrow=nrow,
        normalize=True, value_range=(-1, 1)
    )
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PILImage.fromarray(arr).save(save_path)


def save_attention_map(
    attn_map: torch.Tensor,   # H×W  (values in [0,1])
    save_path: str,
    colormap: str = "hot",
):
    """Save a 2-D attention map as a coloured PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = attn_map.detach().cpu().float().numpy()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr, cmap=colormap, vmin=0, vmax=arr.max())
        ax.axis("off")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except ImportError:
        pass   # matplotlib not installed → skip


def save_inpainting_result(
    original: torch.Tensor,   # 3×H×W
    masked: torch.Tensor,     # 3×H×W
    predicted: torch.Tensor,  # 3×H×W
    save_path: str,
):
    """Save a side-by-side comparison (original | masked | predicted)."""
    from PIL import Image as PILImage

    imgs = [_to_uint8(t) for t in [original, masked, predicted]]
    widths = [i.shape[1] for i in imgs]
    H = imgs[0].shape[0]
    canvas = np.zeros((H, sum(widths), 3), dtype=np.uint8)

    x = 0
    for img in imgs:
        canvas[:, x: x + img.shape[1]] = img
        x += img.shape[1]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PILImage.fromarray(canvas).save(save_path)


def log_training_samples(
    step: int,
    batch: dict,
    predicted: torch.Tensor,
    sample_dir: str,
    num_samples: int = 4,
):
    """Convenience wrapper called from the trainer every N steps."""
    n = min(num_samples, predicted.shape[0])
    save_sample_grid(
        {
            "original":  batch["image"][:n],
            "masked":    batch["masked_image"][:n],
            "predicted": predicted[:n],
        },
        save_path=os.path.join(sample_dir, f"step_{step:07d}.png"),
        nrow=n,
    )
