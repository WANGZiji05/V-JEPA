"""
World Model for L4 MPC experiments.

Wraps frozen V-JEPA / HamJEPA encoders + predictors with:
  - Action adapter: injects action into latent before prediction
  - Reward predictor: maps latent → pole angle (for CEM scoring)
  - Unified predict(z, action) → (z_next, reward)

Both adapters are SMALL (2-layer MLPs, ~50K params each) and trained
on a few thousand CartPole frames collected via random policy.
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ──
_VJEPA_ROOT = os.path.join(os.path.dirname(__file__), '..')
_HAMJEPA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'HamJEPA')
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)
if _HAMJEPA_ROOT not in sys.path:
    sys.path.insert(0, _HAMJEPA_ROOT)


# ============================================================================
# V-JEPA Model Wrapper
# ============================================================================

class VJEPAWorldModel(nn.Module):
    """
    V-JEPA encoder + predictor, adapted for single-frame latent dynamics.

    Encoder: ViT, input [B, C, T, H, W] → output [B, N, D] tokens
    Predictor: transformer, maps (context_tokens, target_tokens, masks) → predicted tokens

    For single-image dynamics, we:
      1. Repeat the frame T=16 times to form a pseudo-clip
      2. Mean-pool encoder tokens → flat vector [B, D]
      3. Use predictor in "self-prediction" mode (context=target=same vector)
         as a latent-space transformation: z → z'
      4. Action adapter: learn offsets added to z before prediction
    """

    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()
        self.device = device

        # Load V-JEPA model
        import src.models.vision_transformer as vit
        import src.models.predictor as vit_pred

        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Detect model variant from checkpoint (encoder or target_encoder)
        if 'encoder' in ckpt:
            enc_state = ckpt['encoder']
        elif 'target_encoder' in ckpt:
            enc_state = ckpt['target_encoder']
        else:
            enc_state = ckpt

        embed_dim = 1024  # fallback
        for k in enc_state:
            if 'pos_embed' in k:
                embed_dim = enc_state[k].shape[-1]
                break

        dim_to_name = {768: 'vit_base', 1024: 'vit_large', 1280: 'vit_huge'}
        model_name = dim_to_name.get(embed_dim, 'vit_large')
        self.embed_dim = embed_dim

        # Create encoder (ViT)
        self.encoder = vit.__dict__[model_name](
            img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
            uniform_power=True, use_sdpa=True, use_SiLU=False, tight_SiLU=False,
        ).to(device).eval()

        # Load encoder weights
        if 'encoder' in ckpt:
            self.encoder.load_state_dict(_clean_state(ckpt['encoder']), strict=False)
        elif 'target_encoder' in ckpt:
            self.encoder.load_state_dict(_clean_state(ckpt['target_encoder']), strict=False)

        for p in self.encoder.parameters():
            p.requires_grad = False

        # Create predictor (detect pred_dim from checkpoint)
        pred_dim = 384  # fallback
        if 'predictor' in ckpt:
            for k in ckpt['predictor']:
                if 'predictor_embed' in k and 'weight' in k:
                    pred_dim = ckpt['predictor'][k].shape[0]
                    break
        self.predictor = vit_pred.VisionTransformerPredictor(
            img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
            embed_dim=embed_dim, predictor_embed_dim=pred_dim,
            depth=12, num_heads=12,
        ).to(device).eval()

        if 'predictor' in ckpt:
            self.predictor.load_state_dict(_clean_state(ckpt['predictor']), strict=False)

        for p in self.predictor.parameters():
            p.requires_grad = False

        # Action adapter: learn 2× offset (for left/right force)
        # CartPole action is 1D continuous; we discretize to K actions for CEM
        self.num_discrete_actions = 5  # [-1, -0.5, 0, 0.5, 1]
        self.action_offsets = nn.Parameter(
            torch.zeros(self.num_discrete_actions, embed_dim)
        )
        nn.init.trunc_normal_(self.action_offsets, std=0.02)

        # Reward predictor: latent → cos(angle) (≈ how upright the pole is)
        self.reward_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),  # cos(angle) ∈ [-1, 1]
        )

    def encode(self, frames):
        """
        frames: [B, T, H, W, C] or [B, C, T, H, W]
        Returns: [B, D] flat latent vector
        """
        if frames.dim() == 5 and frames.shape[-1] == 3:
            # [B, T, H, W, C] → [B, C, T, H, W]
            frames = frames.permute(0, 4, 1, 2, 3)
        elif frames.dim() == 4 and frames.shape[-1] == 3:
            # [T, H, W, C] → [1, C, T, H, W]
            frames = frames.permute(3, 0, 1, 2).unsqueeze(0)

        frames = frames.to(device=self.device, dtype=next(self.encoder.parameters()).dtype)
        with torch.no_grad():
            tokens = self.encoder(frames)  # [B, N, D]
        return tokens.mean(dim=1)  # [B, D]

    def predict(self, z, action_idx):
        """
        Action-conditioned latent dynamics.

        z:          [B, D] current latent state
        action_idx: [B] integer action index (0..num_discrete_actions-1)

        Returns: [B, D] predicted next latent state
        """
        # Inject action as learned offset
        offset = self.action_offsets[action_idx]  # [B, D]
        z_a = z + offset

        # V-JEPA predictor: reshape to [B, 1, D] and pass as context=target
        z_tok = z_a.unsqueeze(1)  # [B, 1, D]
        mask = torch.zeros(z.shape[0], 1, dtype=torch.long, device=z.device)

        with torch.no_grad():
            z_next = self._predictor_forward(z_tok, mask)
        return z_next.squeeze(1)  # [B, D]

    def _predictor_forward(self, z_tok, mask):
        """
        Use V-JEPA predictor in self-prediction mode.
        context = target = z_tok, single-token masks.
        """
        B = z_tok.shape[0]
        device = z_tok.device

        x = self.predictor.predictor_embed(z_tok)  # [B, 1, pred_D]

        # Position embedding for context
        pos = self.predictor.predictor_pos_embed.repeat(B, 1, 1).to(device)
        from src.masks.utils import apply_masks
        x = x + apply_masks(pos, [mask])

        # Target tokens (same as context)
        tgt = self.predictor.predictor_embed(z_tok)
        tgt = self.predictor.diffusion(tgt)
        tgt = tgt + apply_masks(pos, [mask])

        # Concatenate
        x = torch.cat([x, tgt], dim=1)  # [B, 2, pred_D]
        masks = torch.cat([mask, mask], dim=1)

        for blk in self.predictor.predictor_blocks:
            x = blk(x, mask=masks)
        x = self.predictor.predictor_norm(x)
        x = x[:, 1:, :]  # take target output
        x = self.predictor.predictor_proj(x)
        return x  # [B, 1, D]

    def get_reward(self, z):
        """Predict cos(pole_angle) ∈ [-1, 1] from latent."""
        return self.reward_predictor(z)

    def get_trainable_params(self):
        return [self.action_offsets] + list(self.reward_predictor.parameters())


# ============================================================================
# HamJEPA Model Wrapper
# ============================================================================

class HamJEPAWorldModel(nn.Module):
    """
    HamJEPA encoder + Hamiltonian predictor, adapted for action-conditioned dynamics.

    Encoder: ResNet-18, input [B, C, H, W] → output flat vector [B, D]
    Predictor: HamiltonianFlowPredictor, maps z → z' via symplectic integration
    """

    def __init__(self, training_config_path, checkpoint_path, device='cuda'):
        super().__init__()
        self.device = device

        # Load config
        try:
            import yaml
            with open(training_config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        except Exception:
            raise RuntimeError(f"Cannot load training config: {training_config_path}")

        mcfg = cfg['model']
        embed_dim = int(mcfg['embed_dim'])
        split_qp = bool(mcfg.get('split_qp', False))

        # Create encoder (ResNet)
        from eval.models.encoder_resnet import ResNetEncoder
        self.encoder = ResNetEncoder(
            out_dim=embed_dim,
            mode=str(mcfg['encoder_mode']),
            token_layer=str(mcfg.get('token_layer', 'layer3')),
            token_d_f=int(mcfg.get('token_d_f', 32)),
            token_hw=int(mcfg['token_hw']) if mcfg.get('token_hw') else None,
            stem=str(mcfg.get('encoder_stem', 'imagenet')),
            split_qp=split_qp,
        ).to(device).eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

        # Create predictor (Hamiltonian Flow)
        from hamjepa.predictor import HamiltonianFlowPredictor
        hcfg = cfg.get('hjepa', {})
        self.predictor = HamiltonianFlowPredictor(
            state_dim=embed_dim,
            hamiltonian=str(hcfg.get('hamiltonian', 'separable')),
            hidden_dim=int(hcfg.get('hidden_dim', 256)),
            depth=int(hcfg.get('depth', 2)),
            residual_scale=float(hcfg.get('residual_scale', 0.01)),
            base_coeff=float(hcfg.get('base_coeff', 1.0)),
            method=str(hcfg.get('method', 'leapfrog')),
            steps=int(hcfg.get('steps', 2)),
            dt=float(hcfg.get('dt', 0.05)),
            integrate_fp32=True,
        ).to(device).eval()

        for p in self.predictor.parameters():
            p.requires_grad = False

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'encoder' in ckpt:
            self.encoder.load_state_dict(_clean_state(ckpt['encoder']), strict=False)
        if 'predictor' in ckpt:
            self.predictor.load_state_dict(_clean_state(ckpt['predictor']), strict=False)

        self.embed_dim = embed_dim
        self.split_qp = split_qp
        self.q_dim = embed_dim // 2 if split_qp else embed_dim

        # Action adapter: discrete action offsets
        self.num_discrete_actions = 5  # [-1, -0.5, 0, 0.5, 1]
        self.action_offsets = nn.Parameter(
            torch.zeros(self.num_discrete_actions, embed_dim)
        )
        nn.init.trunc_normal_(self.action_offsets, std=0.02)

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def encode(self, frames):
        """
        frames: [B, T, H, W, C] or [B, C, H, W] (single image)
        Returns: [B, D] flat latent vector
        """
        if frames.dim() == 5:
            # [B, T, H, W, C] → take last frame → [B, H, W, C]
            frames = frames[:, -1] if frames.shape[1] > 1 else frames[:, 0]
            frames = frames.permute(0, 3, 1, 2)  # [B, C, H, W]
        elif frames.dim() == 4 and frames.shape[-1] == 3:
            frames = frames.permute(3, 0, 1, 2)  # [H,W,C] → [C,H,W]

        frames = frames.to(device=self.device, dtype=next(self.encoder.parameters()).dtype)
        with torch.no_grad():
            z = self.encoder(frames)  # [B, D]
        return z

    def predict(self, z, action_idx):
        """
        Action-conditioned Hamiltonian flow.

        z:          [B, D]
        action_idx: [B] integer action index

        Returns: [B, D] predicted next latent state
        """
        offset = self.action_offsets[action_idx]  # [B, D]
        z_a = z + offset

        # Hamiltonian predictor needs grad for dV/dq
        with torch.enable_grad():
            z_next = self.predictor(z_a.requires_grad_(True), direction=1)
        return z_next.detach()

    def get_reward(self, z):
        return self.reward_predictor(z)

    def get_trainable_params(self):
        return [self.action_offsets] + list(self.reward_predictor.parameters())


# ============================================================================
# Shared utilities
# ============================================================================

CONTINUOUS_ACTIONS = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)


def continuous_action_to_idx(action):
    """Map continuous action [-1, 1] to nearest discrete index."""
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu()
    action = torch.as_tensor(action, dtype=torch.float32).flatten()
    dist = torch.abs(CONTINUOUS_ACTIONS - action[0])
    return dist.argmin().item()


def idx_to_continuous_action(idx):
    """Map discrete index back to continuous action."""
    return CONTINUOUS_ACTIONS[idx].item()


def _clean_state(state_dict):
    """Remove module/backbone prefixes from state dict keys."""
    new = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('backbone.', '')
        new[k] = v
    return new
