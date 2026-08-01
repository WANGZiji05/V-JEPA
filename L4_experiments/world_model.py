"""
World Model for L4 MPC experiments.

Frozen encoder (V-JEPA ViT / HamJEPA ResNet) + trainable components:
  - DynamicsMLP:   (z, action) → z_next   (replaces frozen predictor)
  - RewardPredictor: z → cos(pole_angle)  (for CEM scoring)

Fair comparison: same MLP architecture, same training data, same protocol.
Only variable = which frozen encoder provides the features.
"""

import os, sys
import torch
import torch.nn as nn

_VJEPA_ROOT = os.path.join(os.path.dirname(__file__), '..')
_HAMJEPA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'HamJEPA')
if _VJEPA_ROOT not in sys.path: sys.path.insert(0, _VJEPA_ROOT)
if _HAMJEPA_ROOT not in sys.path: sys.path.insert(0, _HAMJEPA_ROOT)


# ============================================================================
# Shared Dynamics MLP
# ============================================================================

class DynamicsMLP(nn.Module):
    """Small MLP: (z, action_idx) → z_next."""

    def __init__(self, feature_dim, num_actions=5, hidden_dim=256):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z, action_idx):
        """z: [B, D], action_idx: [B] int """
        a = self.action_embed(action_idx)          # [B, D]
        x = torch.cat([z, a], dim=-1)             # [B, 2D]
        dz = self.net(x)                           # [B, D]
        return z + dz                              # residual: predict delta


# ============================================================================
# Reward Predictor (shared architecture)
# ============================================================================

class RewardPredictor(nn.Module):
    """Small MLP: z → cos(pole_angle) ∈ [-1, 1]."""

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================================
# V-JEPA World Model
# ============================================================================

class VJEPAWorldModel(nn.Module):
    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Detect encoder from checkpoint
        if 'encoder' in ckpt:      enc_state = ckpt['encoder']
        elif 'target_encoder' in ckpt: enc_state = ckpt['target_encoder']
        else:                       enc_state = ckpt

        embed_dim = 1024
        for k in enc_state:
            if 'pos_embed' in k:
                embed_dim = enc_state[k].shape[-1]
                break

        dim_to_name = {768: 'vit_base', 1024: 'vit_large', 1280: 'vit_huge'}
        model_name = dim_to_name.get(embed_dim, 'vit_large')
        self.feature_dim = embed_dim

        import src.models.vision_transformer as vit
        self.encoder = vit.__dict__[model_name](
            img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
            uniform_power=True, use_sdpa=True, use_SiLU=False, tight_SiLU=False,
        ).to(device).eval()
        self.encoder.load_state_dict(_clean(enc_state), strict=False)
        for p in self.encoder.parameters(): p.requires_grad = False

        self.dynamics = DynamicsMLP(embed_dim)
        self.reward = RewardPredictor(embed_dim)
        self.num_discrete_actions = 5

    def encode(self, frames):
        """:param frames: [B, T, H, W, C] or [B, C, T, H, W]"""
        if frames.dim() == 5 and frames.shape[-1] == 3:
            frames = frames.permute(0, 4, 1, 2, 3)
        elif frames.dim() == 4 and frames.shape[-1] == 3:
            frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
        frames = frames.to(device=self.device, dtype=next(self.encoder.parameters()).dtype)
        with torch.no_grad():
            tokens = self.encoder(frames)
        return tokens.mean(dim=1)

    def predict(self, z, action_idx):
        return self.dynamics(z, action_idx)

    def get_reward(self, z):
        return self.reward(z)

    def get_trainable_params(self):
        return list(self.dynamics.parameters()) + list(self.reward.parameters())


# ============================================================================
# HamJEPA World Model
# ============================================================================

class HamJEPAWorldModel(nn.Module):
    def __init__(self, training_config_path, checkpoint_path, device='cuda'):
        super().__init__()
        self.device = device

        import yaml
        with open(training_config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        mcfg = cfg['model']
        embed_dim = int(mcfg['embed_dim'])
        self.feature_dim = embed_dim

        from eval.models.encoder_resnet import ResNetEncoder
        self.encoder = ResNetEncoder(
            out_dim=embed_dim,
            mode=str(mcfg['encoder_mode']),
            token_layer=str(mcfg.get('token_layer', 'layer3')),
            token_d_f=int(mcfg.get('token_d_f', 32)),
            token_hw=int(mcfg['token_hw']) if mcfg.get('token_hw') else None,
            stem=str(mcfg.get('encoder_stem', 'imagenet')),
            split_qp=bool(mcfg.get('split_qp', False)),
        ).to(device).eval()

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'encoder' in ckpt:
            self.encoder.load_state_dict(_clean(ckpt['encoder']), strict=False)
        for p in self.encoder.parameters(): p.requires_grad = False

        self.dynamics = DynamicsMLP(embed_dim)
        self.reward = RewardPredictor(embed_dim)
        self.num_discrete_actions = 5

    def encode(self, frames):
        """:param frames: [B, T, H, W, C] — take last frame"""
        if frames.dim() == 5:
            frames = frames[:, -1] if frames.shape[1] > 1 else frames[:, 0]
            frames = frames.permute(0, 3, 1, 2)
        elif frames.dim() == 4 and frames.shape[-1] == 3:
            frames = frames.permute(3, 0, 1, 2)
        frames = frames.to(device=self.device, dtype=next(self.encoder.parameters()).dtype)
        with torch.no_grad():
            z = self.encoder(frames)
        return z

    def predict(self, z, action_idx):
        return self.dynamics(z, action_idx)

    def get_reward(self, z):
        return self.reward(z)

    def get_trainable_params(self):
        return list(self.dynamics.parameters()) + list(self.reward.parameters())


# ============================================================================
# Shared utilities
# ============================================================================

CONTINUOUS_ACTIONS = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)


def continuous_action_to_idx(action):
    action = torch.as_tensor(action, dtype=torch.float32).flatten()
    return torch.abs(CONTINUOUS_ACTIONS - action[0]).argmin().item()


def idx_to_continuous_action(idx):
    return CONTINUOUS_ACTIONS[idx].item()


def _clean(state_dict):
    return {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
