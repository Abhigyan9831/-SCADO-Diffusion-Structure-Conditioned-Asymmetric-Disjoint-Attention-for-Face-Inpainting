"""
train.py — Entry point.

Usage:
    python train.py
    python train.py --resume checkpoints/last.pt
"""

import argparse
import random
import numpy as np
import torch

from trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True   # faster with fixed input sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cfg",  default="configs/train.yaml")
    parser.add_argument("--model-cfg",  default="configs/model.yaml")
    parser.add_argument("--resume",     default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    trainer = Trainer(args.train_cfg, args.model_cfg)

    set_seed(trainer.tcfg["training"]["seed"])

    start_epoch = 0                                    # ← FIX 3: track start epoch
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    trainer.train(start_epoch=start_epoch)             # ← FIX 4: pass it in


if __name__ == "__main__":
    main()
