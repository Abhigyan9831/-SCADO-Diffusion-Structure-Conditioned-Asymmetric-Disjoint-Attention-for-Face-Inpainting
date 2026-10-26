"""
trainer.py — Training engine for SCADO + Diffusion inpainting.

Features:
  • Mixed precision (fp16) via torch.amp
  • Gradient clipping
  • EMA model
  • Progressive mask curriculum
  • Checkpoint save (best / last / EMA)
  • TensorBoard logging
"""
import numpy as np 
import os
import copy
import math
import yaml
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import InpaintingDataset
from data.mask_generator import MaskGenerator
from models.unet import DiffusionUNet
from models.diffusion import GaussianDiffusion
from losses.losses import InpaintingLoss, PatchGANDiscriminator
from utils.metrics import MetricTracker
from utils.visualization import log_training_samples

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for (name, sp), (_, mp) in zip(
            self.shadow.named_parameters(), model.named_parameters()
        ):
            sp.data = self.decay * sp.data + (1.0 - self.decay) * mp.data

    def state_dict(self):
        return self.shadow.state_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, train_cfg_path: str, model_cfg_path: str):
        with open(train_cfg_path) as f:
            self.tcfg = yaml.safe_load(f)
        with open(model_cfg_path) as f:
            self.mcfg = yaml.safe_load(f)

        self.cfg = {**self.tcfg, **self.mcfg}

        tc = self.tcfg["training"]
        self.epochs     = tc["epochs"]
        self.grad_clip  = tc["gradient_clip"]
        self.log_every  = tc["log_every"]
        self.save_every = tc["save_every"]
        self.val_every  = tc["val_every"]
        self.use_amp    = tc["mixed_precision"]

        paths = self.tcfg["paths"]
        self.ckpt_dir   = Path(paths["checkpoint_dir"])
        self.log_dir    = Path(paths["log_dir"])
        self.sample_dir = Path(paths["sample_dir"])
        for d in [self.ckpt_dir, self.log_dir, self.sample_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self._build_data()
        self._build_models()
        self._build_optimizers()

        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.best_psnr = 0.0
        self.best_fid   = float("inf")    
        self.best_lpips = float("inf")    
        self.best_l1    = float("inf")

        # Initialize tracker once to avoid re-loading LPIPS every validation
        logger.info("Initializing MetricTracker (loading LPIPS/FID models)...")
        self.val_tracker = MetricTracker(device=self.device, compute_fid=True)

    # ──────────────────────────────────────────────────────────────────
    def _build_data(self):
        tc = self.tcfg["training"]
        dc = self.tcfg["data"]
        curriculum = self.tcfg["mask_curriculum"]

        self.mask_gen_train = MaskGenerator(dc["image_size"])
        self.mask_gen_val   = MaskGenerator(dc["image_size"])
        self.mask_gen_val.min_ratio = 0.4
        self.mask_gen_val.max_ratio = 0.5

        self.train_ds = InpaintingDataset(
            root_dir=dc["train_dir"],
            image_size=dc["image_size"],
            num_landmarks=dc["num_landmarks"],
            mask_generator=self.mask_gen_train,
        )
        self.val_ds = InpaintingDataset(
            root_dir=dc["val_dir"],
            image_size=dc["image_size"],
            num_landmarks=dc["num_landmarks"],
            mask_generator=self.mask_gen_val,
        )

        bs = tc["batch_size"]
        self.train_loader = DataLoader(
            self.train_ds, batch_size=bs, shuffle=True,
            num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=bs, shuffle=False,
            num_workers=tc["num_workers"], pin_memory=True,
        )
        self.curriculum = curriculum

    # ──────────────────────────────────────────────────────────────────
    def _build_models(self):
        unet = DiffusionUNet(self.cfg).to(self.device)

        dc = self.mcfg["diffusion"]
        self.diffusion = GaussianDiffusion(
            model=unet,
            num_timesteps=dc["num_timesteps"],
            noise_schedule=dc["noise_schedule"],
            beta_start=dc["beta_start"],
            beta_end=dc["beta_end"],
            sampling_steps=dc["sampling_steps"],
        ).to(self.device)

        self.discriminator = PatchGANDiscriminator(
            in_ch=7,
            base_ch=self.mcfg["discriminator"]["base_channels"],
            num_layers=self.mcfg["discriminator"]["num_layers"],
            use_spectral_norm=self.mcfg["discriminator"]["use_spectral_norm"],
        ).to(self.device)

        lc = self.tcfg["loss"]
        self.criterion = InpaintingLoss(
            lambda_sym=lc["lambda_sym"],
            lambda_perceptual=lc["lambda_perceptual"],
            lambda_adv=lc["lambda_adv"],
        ).to(self.device)

        self.ema = EMA(self.diffusion.model, self.tcfg["training"]["ema_decay"])

        n_params = sum(p.numel() for p in self.diffusion.parameters())
        logger.info(f"Model params: {n_params / 1e6:.1f}M")

    # ──────────────────────────────────────────────────────────────────
    def _build_optimizers(self):
        tc = self.tcfg["training"]
        lr = float(tc["learning_rate"])

        self.opt_g = torch.optim.AdamW(
            self.diffusion.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr * 0.5, betas=(0.9, 0.999)
        )

        total_steps = self.epochs * len(self.train_loader)
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, T_max=total_steps, eta_min=1e-6
        )
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, T_max=total_steps, eta_min=1e-7
        )

    # ──────────────────────────────────────────────────────────────────
    def _to_device(self, batch: dict):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # ──────────────────────────────────────────────────────────────────
    def _train_step(self, batch: dict):
        image      = batch["image"]
        mask       = batch["mask"]
        masked_img = batch["masked_image"]
        landmarks  = batch["landmarks"]

        # ── Generator step ────────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            losses = self.diffusion.compute_loss(image, masked_img, mask, landmarks)
            diff_loss = losses["diffusion_loss"]
            sym_loss  = losses["sym_loss"]
            pred_img  = losses["pred_img"] 

            #torch.cuda.empty_cache()

            # BUG FIX: removed torch.no_grad() — pred_img needs gradients
            # so perceptual and adversarial losses backpropagate into generator
            #t_mid = torch.randint(100, 500, (image.shape[0],), device=self.device)
            #x_t, _ = self.diffusion.q_sample(image, t_mid)
            #noise_pred_vis, _ = self.diffusion.model(x_t, t_mid, masked_img, mask, landmarks)
            #pred_img = self.diffusion.predict_x0(x_t, t_mid, noise_pred_vis).clamp(-1, 1)
            #diff = (pred_img - image).abs().mean().item()
            #self.writer.add_scalar("debug/pred_vs_gt_diff", diff, self.global_step)

            fake_input  = torch.cat([pred_img, masked_img, mask], dim=1)
            fake_logits = self.discriminator(fake_input.detach())

            g_losses = self.criterion.generator_loss(
                pred_img, image, diff_loss, sym_loss, fake_logits
            )
            g_total = g_losses["total"]

        self.opt_g.zero_grad(set_to_none=True)
        self.scaler_g.scale(g_total).backward()
        self.scaler_g.unscale_(self.opt_g)
        nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.grad_clip)
        self.scaler_g.step(self.opt_g)    # BUG FIX: don't check return value
        self.scaler_g.update()
        self.sched_g.step()               # BUG FIX: always step scheduler
        self.ema.update(self.diffusion.model)

        # ── Discriminator step ────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            real_input  = torch.cat([image, masked_img, mask], dim=1)
            real_logits = self.discriminator(real_input)
            fake_logits = self.discriminator(fake_input.detach())
            d_loss = self.criterion.discriminator_loss(real_logits, fake_logits)

        self.opt_d.zero_grad(set_to_none=True)
        self.scaler_d.scale(d_loss).backward()
        self.scaler_d.unscale_(self.opt_d)
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
        self.scaler_d.step(self.opt_d)    # BUG FIX: don't check return value
        self.scaler_d.update()
        self.sched_d.step()               # BUG FIX: always step scheduler

        return {**g_losses, "d_loss": d_loss, "pred": pred_img}

    # ──────────────────────────────────────────────────────────────────
    def _validate(self, epoch: int):
        self.diffusion.eval()
        self.val_tracker.reset()

        l1_vals = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                pred = self.diffusion.ddim_sample(
                    batch["masked_image"], batch["mask"], batch["landmarks"]
                )
                self.val_tracker.update(pred, batch["image"])
                l1 = (pred - batch["image"]).abs().mean().item()
                l1_vals.append(l1)

        metrics = self.val_tracker.compute()
        metrics["l1"] = float(np.mean(l1_vals))
        self.diffusion.train()

        logger.info(
            f"[Val epoch {epoch}] "
            f"PSNR={metrics['psnr']:.2f}  "
            f"SSIM={metrics['ssim']:.4f}  "
            f"FID={metrics['fid']:.2f}  "
            f"LPIPS={metrics.get('lpips', -1):.4f}  "
            f"L1={metrics['l1']:.4f}"
        )
        self.writer.add_scalar("val/psnr",  metrics["psnr"],  epoch)
        self.writer.add_scalar("val/ssim",  metrics["ssim"],  epoch)
        self.writer.add_scalar("val/fid",   metrics["fid"],   epoch)
        self.writer.add_scalar("val/lpips", metrics.get("lpips", -1), epoch)
        self.writer.add_scalar("val/l1",    metrics["l1"],    epoch)
        return metrics

    # ──────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, tag: str):
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save({
                "epoch": epoch,
                "step": self.global_step,
                "diffusion": self.diffusion.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "ema": self.ema.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "best_psnr":  self.best_psnr,
                "best_fid":   self.best_fid,       
                "best_lpips": self.best_lpips,     
                "best_l1":    self.best_l1,        
            }, path)
        logger.info(f"Saved checkpoint: {path}")
        if tag.startswith("epoch_"):
            old_ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
            while len(old_ckpts) > 1:
                oldest = old_ckpts.pop(0)
                oldest.unlink()
                logger.info(f"Deleted: {oldest.name}")

    def load_checkpoint(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.diffusion.load_state_dict(ck["diffusion"])
        self.discriminator.load_state_dict(ck["discriminator"])
        self.opt_g.load_state_dict(ck["opt_g"])
        self.opt_d.load_state_dict(ck["opt_d"])
        self.global_step = ck.get("step", 0)
        self.best_psnr   = ck.get("best_psnr", 0.0)
        self.best_fid   = ck.get("best_fid",   float("inf"))    
        self.best_lpips = ck.get("best_lpips", float("inf"))    
        self.best_l1    = ck.get("best_l1",    float("inf")) 
        return ck.get("epoch", 0)

    # ──────────────────────────────────────────────────────────────────
    def train(self, start_epoch: int = 0):
        logger.info("Starting training ...")

        for epoch in range(start_epoch, self.epochs):
            self.mask_gen_train.update_curriculum(epoch, self.curriculum)
            logger.info(
                f"Epoch {epoch}: mask ratio [{self.mask_gen_train.min_ratio:.0%}, "
                f"{self.mask_gen_train.max_ratio:.0%}]  "
                f"types={self.mask_gen_train.active_types}"
            )

            self.diffusion.train()
            t0 = time.time()

            for batch in self.train_loader:
                batch  = self._to_device(batch)
                result = self._train_step(batch)
                self.global_step += 1

                if self.global_step % self.log_every == 0:
                    self.writer.add_scalar("train/loss_total",      result["total"].item(),      self.global_step)
                    self.writer.add_scalar("train/loss_diffusion",  result["diffusion"].item(),  self.global_step)
                    self.writer.add_scalar("train/loss_sym",        result["sym"].item(),        self.global_step)
                    self.writer.add_scalar("train/loss_perceptual", result["perceptual"].item(), self.global_step)
                    self.writer.add_scalar("train/loss_adv_g",      result["adv_g"].item(),      self.global_step)
                    self.writer.add_scalar("train/loss_adv_d",      result["d_loss"].item(),     self.global_step)

                    elapsed = time.time() - t0
                    logger.info(
                        f"[E{epoch} S{self.global_step}] "
                        f"total={result['total'].item():.4f}  "
                        f"diff={result['diffusion'].item():.4f}  "
                        f"sym={result['sym'].item():.4f}  "
                        f"({elapsed:.1f}s)"
                    )
                    t0 = time.time()

                    log_training_samples(
                        self.global_step, batch,
                        result["pred"],
                        str(self.sample_dir),
                    )

            if epoch % self.val_every == 0:
                metrics = self._validate(epoch)

                if metrics["psnr"] > self.best_psnr:
                    self.best_psnr = metrics["psnr"]
                    self._save_checkpoint(epoch, "best_psnr")
                    logger.info(f"New best PSNR: {self.best_psnr:.2f}")

                if metrics["fid"] > 0 and metrics["fid"] < self.best_fid:
                    self.best_fid = metrics["fid"]
                    self._save_checkpoint(epoch, "best_fid")
                    logger.info(f"New best FID: {self.best_fid:.2f}")

                if metrics.get("lpips", -1) > 0 and metrics["lpips"] < self.best_lpips:
                    self.best_lpips = metrics["lpips"]
                    self._save_checkpoint(epoch, "best_lpips")
                    logger.info(f"New best LPIPS: {self.best_lpips:.4f}")

                if metrics["l1"] < self.best_l1:
                    self.best_l1 = metrics["l1"]
                    self._save_checkpoint(epoch, "best_l1")
                    logger.info(f"New best L1: {self.best_l1:.4f}")

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch:04d}")

            self._save_checkpoint(epoch, "last")

        logger.info("Training complete.")
        self.writer.close()