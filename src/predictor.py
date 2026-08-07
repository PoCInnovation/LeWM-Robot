"""
Action-conditioned predictor (world model dynamics).

Architecture inspirée de DINO-WM :
    Input  : z_t (B, N, dim) latent fusionné + action_t (B, action_dim)
    Output : z_{t+1} (B, N, dim) prédiction du prochain latent

Le predictor est un transformer compact qui voit l'action comme un token
supplémentaire et apprend la dynamique action -> conséquence dans le latent.

Utilisé:
    - Training (Phase A) : sur les données Isaac sim
    - LoRA fine-tune (Phase B) : ajout d'adapters sur les 40 démos réelles
    - Inference : appelé en boucle par le planner CEM
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PredictorConfig:
    """Configuration du predictor."""
    embed_dim: int = 768                # doit matcher l'output de DINOv3
    action_dim: int = 6                 # 6 servos SO-101
    n_layers: int = 6
    n_heads: int = 12
    ffn_dim: Optional[int] = None       # None = 4x embed_dim
    dropout: float = 0.1
    context_length: Optional[int] = None  # si None, calcul auto
    n_action_tokens: int = 1            # nombre de tokens pour encoder l'action
    predict_delta: bool = True          # sortie = z_t + head(trunk).
    # Pourquoi : la LayerNorm finale renormalise chaque token → la sortie ne
    # peut jamais coller exactement à des latents cibles NON normalisés
    # (plancher de MSE incompressible, vérifié empiriquement). En résiduel
    # avec une head initialisée à zéro, l'identité est exacte à l'init et le
    # modèle n'apprend que la dynamique (le delta).

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.embed_dim


class ActionEmbedding(nn.Module):
    """
    Projette une action (action_dim,) vers (n_action_tokens, embed_dim).
    """
    def __init__(self, action_dim: int, embed_dim: int, n_tokens: int = 1):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim

        if n_tokens == 1:
            self.proj = nn.Linear(action_dim, embed_dim)
        else:
            # Projection avec une dimension supplémentaire pour les tokens
            self.proj = nn.Linear(action_dim, n_tokens * embed_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        # action : (B, action_dim)
        proj = self.proj(action)  # (B, embed_dim ou n_tokens * embed_dim)
        if self.n_tokens == 1:
            return proj.unsqueeze(1)  # (B, 1, embed_dim)
        else:
            return proj.view(-1, self.n_tokens, self.embed_dim)


class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention multi-tête avec des nn.Linear explicites (Q, K, V, O).

    Implémentation custom (au lieu de nn.MultiheadAttention) pour que
    LoRA puisse injecter ses adapters dans chaque projection séparément.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim doit être divisible par n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Quatre Linear séparées (LoRA-friendly)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        k = self.k_proj(x).reshape(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, Dh).transpose(1, 2)

        # Scaled dot-product attention via PyTorch fused
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        # (B, H, N, Dh) -> (B, N, D)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        out = self.o_proj(attn)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """Bloc transformer standard (self-attention + FFN), LoRA-friendly."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm style
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class WorldModelPredictor(nn.Module):
    """
    Predictor de world model action-conditionné.

    Inspired by DINO-WM : transformer qui prend (latent_patches + action_token)
    et prédit les latent_patches au temps suivant.

    Forward:
        z_t    : (B, N, dim) latent fusionné (sortie de la fusion multi-cam)
        action : (B, action_dim)

    Returns:
        z_t+1  : (B, N, dim) prédiction du prochain latent
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config

        # Embedding d'action
        self.action_emb = ActionEmbedding(
            action_dim=config.action_dim,
            embed_dim=config.embed_dim,
            n_tokens=config.n_action_tokens,
        )

        # Positional encoding (appris) — pour permettre au model de distinguer
        # les positions dans la séquence (incluant le/les token(s) action)
        # On utilisera une dim large par défaut, ajustable si besoin
        max_pos = config.context_length or 1024
        self.pos_emb = nn.Parameter(torch.randn(1, max_pos, config.embed_dim) * 0.02)

        # Blocs transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                n_heads=config.n_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

        # Head de sortie (mode résiduel) : z_t+1 = z_t + head(trunk)
        self.out_head: Optional[nn.Linear] = None
        if config.predict_delta:
            self.out_head = nn.Linear(config.embed_dim, config.embed_dim)

        # Init weights
        self.apply(self._init_weights)

        # Head à ZÉRO : à l'init, le predictor est exactement l'identité
        if self.out_head is not None:
            nn.init.zeros_(self.out_head.weight)
            nn.init.zeros_(self.out_head.bias)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t    : (B, N, embed_dim) latent au temps t (déjà fusionné cross-cam)
            action : (B, action_dim) action prise au temps t

        Returns:
            z_t_plus_1 : (B, N, embed_dim) prédiction au temps t+1
        """
        B, N, D = z_t.shape
        n_act = self.config.n_action_tokens

        # Embedding de l'action -> tokens
        action_tokens = self.action_emb(action)  # (B, n_act, D)

        # Concat : [action_token(s), latent_patches]
        x = torch.cat([action_tokens, z_t], dim=1)  # (B, n_act + N, D)

        # Ajout positional encoding
        seq_len = x.shape[1]
        assert seq_len <= self.pos_emb.shape[1], (
            f"Sequence length {seq_len} > max_pos {self.pos_emb.shape[1]}. "
            "Augmente context_length dans PredictorConfig."
        )
        x = x + self.pos_emb[:, :seq_len]

        # Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # On retire les tokens d'action, on ne garde que les patches
        out = x[:, n_act:]  # (B, N, D)

        if self.out_head is not None:
            # Résiduel : le trunk prédit un DELTA ajouté au latent d'entrée
            return z_t + self.out_head(out)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def rollout(predictor: WorldModelPredictor,
            z_start: torch.Tensor,
            action_sequence: torch.Tensor) -> torch.Tensor:
    """
    Déroule le predictor sur une séquence d'actions.

    Utilisé par le CEM planner pour simuler des trajectoires candidates.

    Args:
        z_start         : (B, N, dim) état latent de départ
        action_sequence : (B, T, action_dim) séquence d'actions à appliquer

    Returns:
        trajectory : (B, T+1, N, dim) — z_0, z_1, ..., z_T
    """
    T = action_sequence.shape[1]
    z = z_start
    trajectory = [z]
    for t in range(T):
        z = predictor(z, action_sequence[:, t, :])
        trajectory.append(z)
    return torch.stack(trajectory, dim=1)  # (B, T+1, N, dim)


if __name__ == "__main__":
    # Sanity check
    config = PredictorConfig(
        embed_dim=768,
        action_dim=6,
        n_layers=6,
        n_heads=12,
    )
    predictor = WorldModelPredictor(config)

    # Test forward simple
    B, N = 2, 392  # 392 = 2 caméras × 196 patches (cas concat fusion)
    z_t = torch.randn(B, N, config.embed_dim)
    action = torch.randn(B, config.action_dim)

    z_t1 = predictor(z_t, action)
    print(f"Input  : z_t={tuple(z_t.shape)}, action={tuple(action.shape)}")
    print(f"Output : z_t1={tuple(z_t1.shape)}")
    print(f"Predictor params : {predictor.count_parameters():,}")

    # Test rollout
    print("\nTest rollout :")
    actions = torch.randn(B, 10, config.action_dim)  # 10 steps
    traj = rollout(predictor, z_t, actions)
    print(f"Trajectory shape : {tuple(traj.shape)} "
          f"(expected: ({B}, 11, {N}, {config.embed_dim}))")

    print("\n[OK] WorldModelPredictor fonctionne.")
