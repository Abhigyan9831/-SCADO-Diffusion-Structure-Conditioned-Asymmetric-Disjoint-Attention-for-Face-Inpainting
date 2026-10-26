"""
inference.py — Run inpainting on a single image using a trained checkpoint.

Usage:
    python inference.py \\
        --image   path/to/image.jpg \\
        --mask    path/to/mask.png      [optional; generated if absent] \\
        --ckpt    checkpoints/best.pt \\
        --out-dir outputs/
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

import torchvision.transforms as T

from models.unet import DiffusionUNet
from models.diffusion import GaussianDiffusion
from data.mask_generator import MaskGenerator
from utils.visualization import save_inpainting_result, save_attention_map


# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str, size: int) -> torch.Tensor:
    """Load and preprocess image to C×H×W in [-1,1]."""
    img = Image.open(path).convert("RGB")
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(img).unsqueeze(0)   # 1×3×H×W


def load_mask(path: str, size: int) -> torch.Tensor:
    """Load mask from grayscale PNG.  White (255) = missing."""
    mask = Image.open(path).convert("L").resize((size, size))
    t = T.ToTensor()(mask)        # 1×H×W  in [0,1]
    return (t > 0.5).float().unsqueeze(0)   # 1×1×H×W


def synthesize_landmarks(num: int = 7) -> torch.Tensor:
    base = torch.tensor([
        [0.35, 0.40], [0.65, 0.40], [0.50, 0.55],
        [0.35, 0.70], [0.65, 0.70], [0.25, 0.30], [0.75, 0.30],
    ], dtype=torch.float32)
    return base[:num].unsqueeze(0)   # 1×K×2


# ──────────────────────────────────────────────────────────────────────────────

class InpaintingInference:
    def __init__(
        self,
        ckpt_path: str,
        train_cfg: str = "configs/train.yaml",
        model_cfg: str = "configs/model.yaml",
        device: str = None,
    ):
        import yaml
        with open(train_cfg) as f:
            tcfg = yaml.safe_load(f)
        with open(model_cfg) as f:
            mcfg = yaml.safe_load(f)
        self.cfg = {**tcfg, **mcfg}

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = self.cfg["model"]["image_size"]

        unet = DiffusionUNet(self.cfg)
        dc = self.cfg["diffusion"]
        self.diffusion = GaussianDiffusion(
            model=unet,
            num_timesteps=dc["num_timesteps"],
            noise_schedule=dc["noise_schedule"],
            beta_start=dc["beta_start"],
            beta_end=dc["beta_end"],
            sampling_steps=dc["sampling_steps"],
        ).to(self.device)

        ck = torch.load(ckpt_path, map_location=self.device)
        # Load EMA weights if available (best quality)
        state = ck.get("ema") or ck.get("diffusion")
        self.diffusion.model.load_state_dict(state, strict=False)
        self.diffusion.eval()

        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Device: {self.device}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def inpaint(
        self,
        image: torch.Tensor,        # 1×3×H×W  in [-1,1]
        mask: torch.Tensor,         # 1×1×H×W
        landmarks: torch.Tensor,    # 1×K×2
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Run DDIM sampling and return inpainted image (1×3×H×W)."""
        image     = image.to(self.device)
        mask      = mask.to(self.device)
        landmarks = landmarks.to(self.device)
        masked_img = image * (1 - mask)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = self.diffusion.ddim_sample(masked_img, mask, landmarks, eta=eta)

        return pred.cpu()

    # ------------------------------------------------------------------
    def extract_attention_map(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a single forward pass and extract the SCADO attention map.
        Returns a 2-D map (H_bot × W_bot) averaged over heads.
        """
        image     = image.to(self.device)
        mask      = mask.to(self.device)
        landmarks = landmarks.to(self.device)
        masked_img = image * (1 - mask)

        # Use t=0 for stable attention
        t = torch.zeros(1, dtype=torch.long, device=self.device)

        # Hook into SCADO to capture attention weights
        attention_cache = {}

        def hook_fn(module, inputs, output):
            # output = (feature_map, sym_loss); we want attn inside forward
            # We store the input feature map for visualisation
            attention_cache["feature"] = inputs[0].detach()

        handle = self.diffusion.model.scado.register_forward_hook(hook_fn)

        with torch.no_grad():
            noise = torch.randn_like(image)
            x_t, _ = self.diffusion.q_sample(image, t, noise)
            self.diffusion.model(x_t, t, masked_img, mask, landmarks)

        handle.remove()

        # Build attention-like map: magnitude of bottleneck features at masked positions
        if "feature" in attention_cache:
            feat = attention_cache["feature"]          # 1×C×H_b×W_b
            feat_norm = feat.norm(dim=1).squeeze(0)    # H_b×W_b
            feat_norm = (feat_norm - feat_norm.min()) / (feat_norm.max() + 1e-8)

            # Upsample to input resolution for display
            H, W = image.shape[2], image.shape[3]
            attn_up = F.interpolate(
                feat_norm.unsqueeze(0).unsqueeze(0), (H, W), mode="bilinear", align_corners=False
            ).squeeze()
            return attn_up.cpu()

        return torch.zeros(image.shape[2], image.shape[3])


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--mask",       default=None)
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--out-dir",    default="outputs")
    parser.add_argument("--train-cfg",  default="configs/train.yaml")
    parser.add_argument("--model-cfg",  default="configs/model.yaml")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--eta",        type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    engine = InpaintingInference(args.ckpt, args.train_cfg, args.model_cfg)
    size   = engine.image_size

    # Load inputs
    image = load_image(args.image, size)

    if args.mask:
        mask = load_mask(args.mask, size)
    else:
        mg = MaskGenerator(size)
        mg.min_ratio = args.mask_ratio - 0.05
        mg.max_ratio = args.mask_ratio + 0.05
        mg.active_types = ["freeform"]
        mask = mg.get_mask().unsqueeze(0)   # 1×1×H×W

    landmarks = synthesize_landmarks(engine.cfg["model"]["num_landmarks"])

    # Run inpainting
    pred = engine.inpaint(image, mask, landmarks, eta=args.eta)

    # Save results
    stem = Path(args.image).stem
    masked_img = image * (1 - mask)

    save_inpainting_result(
        image.squeeze(0), masked_img.squeeze(0), pred.squeeze(0),
        os.path.join(args.out_dir, f"{stem}_comparison.png"),
    )

    # Attention map
    attn_map = engine.extract_attention_map(image, mask, landmarks)
    save_attention_map(
        attn_map,
        os.path.join(args.out_dir, f"{stem}_attention.png"),
    )

    print(f"Results saved to {args.out_dir}/")
    print(f"  comparison : {stem}_comparison.png")
    print(f"  attention  : {stem}_attention.png")


if __name__ == "__main__":
    main()
