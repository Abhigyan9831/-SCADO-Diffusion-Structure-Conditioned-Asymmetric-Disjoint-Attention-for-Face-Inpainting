"""
landmark_encoder.py — Encodes landmark coordinates into embedding vectors
that can be injected into the U-Net as conditioning signals.

Architecture:
  coords (B×K×2) → MLP → per-landmark embed (B×K×d)
                       → flatten → linear → global embed (B×d)
"""

import torch
import torch.nn as nn


class LandmarkEncoder(nn.Module):
    """
    Maps landmark coordinates to a conditioning embedding.

    Args:
        num_landmarks : number of structural landmarks (K)
        embed_dim     : output embedding dimension (matches U-Net time-embed)
    """

    def __init__(self, num_landmarks: int = 7, embed_dim: int = 256):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.embed_dim = embed_dim

        # Per-landmark MLP: 2 → hidden → local_d
        local_d = embed_dim // 2
        self.per_lm_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, local_d),
            nn.SiLU(),
        )

        # Global aggregation: K * local_d → embed_dim
        self.global_mlp = nn.Sequential(
            nn.Linear(num_landmarks * local_d, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            landmarks: B × K × 2  (normalised to [0,1])
        Returns:
            embed: B × embed_dim
        """
        B = landmarks.shape[0]

        # Per-landmark embedding
        lm_embed = self.per_lm_mlp(landmarks)   # B×K×local_d
        lm_flat = lm_embed.view(B, -1)           # B×(K*local_d)

        # Global embedding
        return self.global_mlp(lm_flat)          # B×embed_dim
