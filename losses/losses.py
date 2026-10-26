"""
losses.py — All loss components for SCADO + Diffusion inpainting.

L_total = L_recon + λ_sym * L_sym + λ_perceptual * L_perceptual + λ_adv * L_adv

Components:
  L_recon      : L1 + diffusion noise MSE
  L_sym        : SCADO symmetry loss (from attention module)
  L_perceptual : VGG-16 feature matching
  L_adv        : PatchGAN discriminator (hinge loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ══════════════════════════════════════════════════════════════════════════════
# Perceptual Loss  (VGG-16)
# ══════════════════════════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    Computes L1 feature matching at selected VGG-16 layers.
    VGG weights are frozen.
    """

    LAYER_INDICES = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 16,
        "relu4_3": 23,
    }

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(pretrained=True)
        features = vgg.features

        # Split into sub-networks at each target layer
        self.blocks = nn.ModuleList()
        prev = 0
        for name, idx in sorted(self.LAYER_INDICES.items(), key=lambda x: x[1]):
            self.blocks.append(nn.Sequential(*list(features.children())[prev:idx + 1]))
            prev = idx + 1

        # Freeze
        for p in self.parameters():
            p.requires_grad_(False)

        # Normalisation (ImageNet stats, input expected in [-1,1])
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1,1] to ImageNet-normalised [0,1]."""
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalise(pred)
        target_n = self._normalise(target)

        loss = pred.new_zeros(1).squeeze()
        p, t = pred_n, target_n
        for block in self.blocks:
            p = block(p)
            t = block(t)
            loss = loss + F.l1_loss(p, t.detach())

        return loss


# ══════════════════════════════════════════════════════════════════════════════
# PatchGAN Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator with spectral normalisation.
    Input: concatenated [predicted/real image, masked_image, mask]  → 7 channels
    """

    def __init__(self, in_ch: int = 7, base_ch: int = 64, num_layers: int = 3,
                 use_spectral_norm: bool = True):
        super().__init__()

        def norm_conv(ic, oc, stride):
            conv = nn.Conv2d(ic, oc, 4, stride=stride, padding=1, bias=False)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            return conv

        layers = [norm_conv(in_ch, base_ch, 2), nn.LeakyReLU(0.2, inplace=True)]
        ch = base_ch
        for i in range(1, num_layers):
            out_ch = min(ch * 2, 512)
            layers += [
                norm_conv(ch, out_ch, 2 if i < num_layers - 1 else 1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch

        # Final patch prediction
        layers.append(norm_conv(ch, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Hinge Adversarial Loss
# ══════════════════════════════════════════════════════════════════════════════

def adversarial_loss_discriminator(real_logits, fake_logits):
    """Hinge loss for D."""
    d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    return d_loss


def adversarial_loss_generator(fake_logits):
    """Hinge loss for G."""
    return -fake_logits.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Combined Loss
# ══════════════════════════════════════════════════════════════════════════════

class InpaintingLoss(nn.Module):
    """
    Aggregates all losses:

        L_total = L_diffusion + L_l1
                + λ_sym  * L_sym
                + λ_perc * L_perceptual
                + λ_adv  * L_adv_gen
    """

    def __init__(self, lambda_sym=0.1, lambda_perceptual=0.1, lambda_adv=0.01):
        super().__init__()
        self.lambda_sym = lambda_sym
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv
        self.perceptual = PerceptualLoss()

    def generator_loss(
        self,
        pred: torch.Tensor,           # B×3×H×W  predicted image
        target: torch.Tensor,         # B×3×H×W  ground truth
        diffusion_loss: torch.Tensor,
        sym_loss: torch.Tensor,
        fake_logits: torch.Tensor = None,
    ) -> dict:
        l1 = F.l1_loss(pred, target)
        perc = self.perceptual(pred, target)

        total = diffusion_loss + l1
        total = total + self.lambda_sym * sym_loss
        total = total + self.lambda_perceptual * perc

        if fake_logits is not None:
            adv = adversarial_loss_generator(fake_logits)
            total = total + self.lambda_adv * adv
        else:
            adv = torch.zeros(1)

        return {
            "total": total,
            "diffusion": diffusion_loss,
            "l1": l1,
            "sym": sym_loss,
            "perceptual": perc,
            "adv_g": adv,
        }

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        return adversarial_loss_discriminator(real_logits, fake_logits)
