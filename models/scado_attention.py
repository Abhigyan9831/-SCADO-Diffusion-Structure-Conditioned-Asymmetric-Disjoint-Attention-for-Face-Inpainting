"""
scado_attention.py — Structure-Conditioned Asymmetric Disjoint Attention (SCADO)

Mathematical formulation:
─────────────────────────────────────────────────────────────────────────────

1. Feature map:  f ∈ R^{C × H × W}

2. Disjoint split based on binary mask M (1=masked, 0=visible):
       Ω_m  = {i : M_i = 1}   (masked positions)
       Ω_v  = {i : M_i = 0}   (visible positions)

3. Queries from masked, Keys/Values from visible:
       Q = W_Q · f[Ω_m]       shape: |Ω_m| × d
       K = W_K · f[Ω_v]       shape: |Ω_v| × d
       V = W_V · f[Ω_v]       shape: |Ω_v| × d

4. Landmark bias:
       Bias_ij = Σ_k α_k · H_k(i) · H_k(j)
   where H_k is the k-th Gaussian landmark heatmap, α_k is a learned weight.

5. Final attention:
       Ã = softmax(Q K^T / sqrt(d) + Bias)
       out[Ω_m] = Ã · V

6. Symmetry loss (horizontal mirror):
       L_sym = Σ_i ||A_i - A_{mirror(i)}||^2

7. Graph Attention for dynamic α_k update:
       e_kl = LeakyReLU(a^T [W_g h_k || W_g h_l])
       h_k' = σ( Σ_l softmax(e_kl) · W_g h_l )
       α_k  = MLP(h_k')

─────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Landmark Heatmap Generator
# ══════════════════════════════════════════════════════════════════════════════

class LandmarkHeatmapEncoder(nn.Module):
    """
    Converts landmark coordinates (x,y ∈ [0,1]) to Gaussian heatmaps.

    H_k(i) = exp( -||p_i - l_k||^2 / (2σ^2) )
    where p_i is the spatial position of pixel i and l_k is landmark k.
    """

    def __init__(self, num_landmarks: int = 7, sigma: float = 5.0):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.sigma = sigma

    def forward(
        self,
        landmarks: torch.Tensor,   # B × num_landmarks × 2  (x,y in [0,1])
        H: int,
        W: int,
    ) -> torch.Tensor:             # B × num_landmarks × H × W
        """
        Build Gaussian heatmaps at the spatial resolution H×W.
        """
        B = landmarks.shape[0]
        device = landmarks.device

        # Grid of pixel positions: (H×W, 2)
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs)  # H×W each
        grid = torch.stack([grid_x, grid_y], dim=-1)            # H×W×2
        grid = grid.view(1, 1, H * W, 2)                        # 1×1×HW×2

        # Landmarks: B×K×1×2
        lm = landmarks.unsqueeze(2)

        # Squared distance: B×K×HW
        diff = grid - lm          # B×K×HW×2
        dist2 = (diff ** 2).sum(-1)

        heatmaps = torch.exp(-dist2 / (2.0 * self.sigma ** 2))  # B×K×HW
        return heatmaps.view(B, self.num_landmarks, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Graph Attention Network for Landmark Weights α_k
# ══════════════════════════════════════════════════════════════════════════════

class LandmarkGraphAttention(nn.Module):
    """
    GAT over K landmark nodes to produce per-landmark weights α_k.

    Each landmark starts with a learned embedding h_k ∈ R^{d_g}.
    GAT update:
        e_kl = LeakyReLU( a^T [W_g h_k || W_g h_l] )
        h_k' = σ( Σ_l softmax(e_kl) · W_g h_l )
        α_k  = MLP(h_k') ∈ R^+
    """

    def __init__(self, num_landmarks: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.K = num_landmarks

        # Shared node transform
        self.W_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Attention scoring vector: acts on concatenated [W_g h_k || W_g h_l]
        self.attn_vec = nn.Parameter(torch.randn(2 * hidden_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = nn.ELU()

        # Map updated node feature → scalar weight
        self.alpha_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),   # ensure α_k > 0
        )

        # Landmark embeddings initialised per-node
        self.node_embed = nn.Embedding(num_landmarks, hidden_dim)

    def forward(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Returns α ∈ R^{B × K}  (positive, sum-normalised across K).
        """
        # Node features: K × hidden_dim
        h = self.node_embed.weight            # K × d_g
        Wh = self.W_g(h)                      # K × d_g

        # Pairwise attention scores: K × K
        # e_kl = LeakyReLU( a^T [Wh_k || Wh_l] )
        Wh_i = Wh.unsqueeze(1).expand(-1, self.K, -1)   # K×K×d_g
        Wh_j = Wh.unsqueeze(0).expand(self.K, -1, -1)   # K×K×d_g
        concat = torch.cat([Wh_i, Wh_j], dim=-1)        # K×K×2d_g
        e = self.leaky_relu((concat * self.attn_vec).sum(-1))  # K×K

        # Softmax over neighbours (full graph — no mask)
        att = F.softmax(e, dim=-1)            # K×K

        # Aggregate:  h_k' = σ( Σ_l att_kl · Wh_l )
        h_prime = self.activation(att @ Wh)  # K×d_g

        # Per-node scalar weight
        alpha = self.alpha_mlp(h_prime).squeeze(-1)      # K
        alpha = alpha / (alpha.sum() + 1e-8)              # normalise

        # Broadcast to batch
        return alpha.unsqueeze(0).expand(B, -1)           # B×K


# ══════════════════════════════════════════════════════════════════════════════
# SCADO Attention Block
# ══════════════════════════════════════════════════════════════════════════════

class SCADOAttention(nn.Module):
    """
    Full SCADO block.

    Inputs
    ------
    x         : B × C × H × W  (feature map from encoder)
    mask      : B × 1 × H × W  (1=masked, 0=visible)
    landmarks : B × K × 2      (normalised coordinates)
    heatmaps  : B × K × H × W  (pre-computed Gaussian heatmaps)

    Output
    ------
    out       : B × C × H × W  (refined feature map)
    sym_loss  : scalar          (symmetry regularisation loss)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_landmarks: int = 7,
        landmark_sigma: float = 5.0,
        graph_hidden_dim: int = 128,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.num_landmarks = num_landmarks

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Landmark components
        self.heatmap_encoder = LandmarkHeatmapEncoder(num_landmarks, landmark_sigma)
        self.graph_attn = LandmarkGraphAttention(num_landmarks, graph_hidden_dim)

        # Project heatmap bias to num_heads (one per head)
        self.bias_proj = nn.Linear(num_landmarks, num_heads, bias=False)

        # Layer norms
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def _build_landmark_bias(
        self,
        heatmaps: torch.Tensor,    # B×K×H×W
        alpha: torch.Tensor,       # B×K
        mask: torch.Tensor,        # B×1×H×W
    ):
        """
        Compute the landmark attention bias:
            Bias_ij = Σ_k α_k · H_k(i) · H_k(j)

        We only need  Bias[Ω_m, Ω_v]  (query=masked, key=visible).

        Returns bias of shape B × num_heads × |Ω_m| × |Ω_v|
        (note: varies per sample; we return a list of tensors).
        """
        B, K, H, W = heatmaps.shape
        mask_flat = mask.view(B, H * W)       # B×HW  (1=masked)
        hm_flat = heatmaps.view(B, K, H * W)  # B×K×HW

        biases = []
        for b in range(B):
            m_idx = mask_flat[b].nonzero(as_tuple=True)[0]    # |Ω_m|
            v_idx = (1 - mask_flat[b]).nonzero(as_tuple=True)[0]  # |Ω_v|

            # H_k values at masked/visible positions: K×|Ω_m| and K×|Ω_v|
            Hm = hm_flat[b][:, m_idx]   # K×|Ω_m|
            Hv = hm_flat[b][:, v_idx]   # K×|Ω_v|

            # Weighted outer product: Σ_k α_k · H_k(m) · H_k(v)
            # shape: |Ω_m|×|Ω_v| per landmark, summed over K
            a = alpha[b]  # K
            # einsum: k, k m, k v -> m v
            bias_mv = torch.einsum("k,km,kv->mv", a, Hm, Hv)  # |Ω_m|×|Ω_v|

            # Project to num_heads via the bias_proj (using landmark dim)
            # We replicate the scalar bias across heads (simple approach)
            bias_mv = bias_mv.unsqueeze(0).expand(self.num_heads, -1, -1)  # H×|Ω_m|×|Ω_v|
            biases.append(bias_mv)

        return biases  # list of B tensors, each H×|Ω_m|×|Ω_v|

    # ------------------------------------------------------------------
    def _compute_symmetry_loss(
        self,
        attn_weights: torch.Tensor,  # |Ω_m|×|Ω_v|
        m_idx: torch.Tensor,
        v_idx: torch.Tensor,
        W: int,
    ) -> torch.Tensor:
        """
        L_sym = Σ_i ||A_i - A_{mirror(i)}||^2

        The horizontal mirror of position p = (row, col) in a W-wide image
        is (row, W-1-col). We find pairs where both p and mirror(p) ∈ Ω_m
        and compute the L2 distance between their attention rows.
        """
        if attn_weights.shape[0] < 2:
            return attn_weights.new_zeros(1).squeeze()

        # Row indices within the masked set that have a mirror also in Ω_m
        m_set = set(m_idx.tolist())
        m_list = m_idx.tolist()
        sym_loss = attn_weights.new_zeros(1).squeeze()
        count = 0

        for local_i, global_i in enumerate(m_list):
            col = global_i % W
            mirror_col = W - 1 - col
            mirror_global = global_i - col + mirror_col
            if mirror_global in m_set and mirror_global != global_i:
                local_mirror = m_list.index(mirror_global)
                diff = attn_weights[local_i] - attn_weights[local_mirror]
                sym_loss = sym_loss + (diff ** 2).mean()
                count += 1

        return sym_loss / max(count, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,          # B×C×H×W
        mask: torch.Tensor,       # B×1×H×W
        landmarks: torch.Tensor,  # B×K×2
    ):
        B, C, H, W = x.shape

        # Build heatmaps and dynamic landmark weights
        heatmaps = self.heatmap_encoder(landmarks, H, W)  # B×K×H×W
        alpha = self.graph_attn(B, x.device)               # B×K

        # Pre-compute landmark bias for each sample
        lm_biases = self._build_landmark_bias(heatmaps, alpha, mask)

        # Flatten spatial dimensions
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # B×HW×C
        x_flat = self.norm(x_flat)
        mask_flat = mask.view(B, H * W)                  # B×HW

        out = x.clone()  # residual base; we fill in Ω_m positions
        sym_loss = x.new_zeros(1).squeeze()

        for b in range(B):
            m_idx = mask_flat[b].nonzero(as_tuple=True)[0]   # |Ω_m|
            v_idx = (1 - mask_flat[b]).nonzero(as_tuple=True)[0]  # |Ω_v|

            if len(m_idx) == 0 or len(v_idx) == 0:
                continue

            # Disjoint feature extraction
            f_mask = x_flat[b][m_idx]   # |Ω_m|×C
            f_vis = x_flat[b][v_idx]    # |Ω_v|×C

            # Q K V projections
            Q = self.W_Q(f_mask)        # |Ω_m|×d
            K = self.W_K(f_vis)         # |Ω_v|×d
            V = self.W_V(f_vis)         # |Ω_v|×d

            # Multi-head split
            def split_heads(t, n):
                # t: N×d  → n×N×(d/n)
                N = t.shape[0]
                return t.view(N, n, self.head_dim).permute(1, 0, 2)

            Q_ = split_heads(Q, self.num_heads)   # H×|Ω_m|×d_h
            K_ = split_heads(K, self.num_heads)   # H×|Ω_v|×d_h
            V_ = split_heads(V, self.num_heads)   # H×|Ω_v|×d_h

            # Scaled dot-product + landmark bias
            # A_ij = QK^T / sqrt(d) + Bias_ij
            raw = torch.bmm(Q_, K_.transpose(1, 2)) / self.scale  # H×|Ω_m|×|Ω_v|
            raw = raw + lm_biases[b]                               # broadcast bias

            attn = F.softmax(raw, dim=-1)           # H×|Ω_m|×|Ω_v|

            # Symmetry loss (on average attention across heads)
            avg_attn = attn.mean(0)                 # |Ω_m|×|Ω_v|
            sym_loss = sym_loss + self._compute_symmetry_loss(avg_attn, m_idx, v_idx, W)

            # Aggregate values
            agg = torch.bmm(attn, V_)               # H×|Ω_m|×d_h
            agg = agg.permute(1, 0, 2).contiguous().view(len(m_idx), C)  # |Ω_m|×C

            # Output projection + residual for masked positions
            agg = self.out_proj(agg)
            out_flat = out[b].view(C, H * W).permute(1, 0)  # HW×C
            out_flat[m_idx] = out_flat[m_idx] + agg
            out[b] = out_flat.permute(1, 0).view(C, H, W)

        sym_loss = sym_loss / B
        return out, sym_loss
