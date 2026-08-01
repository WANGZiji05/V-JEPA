"""
Train action adapter + reward predictor on CartPole data collected via random policy.

The adapters are SMALL (action_offsets: 5×D params, reward_predictor: ~50K params).
Training on 1000-2000 frames takes ~2 minutes on GPU.
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_wrapper import CartPolePixelEnv, collect_random_data
from world_model import (
    VJEPAWorldModel, HamJEPAWorldModel,
    continuous_action_to_idx, idx_to_continuous_action,
)


def train_adapter(world_model, data, epochs=200, batch_size=128, lr=1e-3, device='cuda'):
    """
    Train action adapter + reward predictor.

    data: dict with keys 'obs', 'action', 'next_obs', 'reward'
          obs: [N, T, H, W, C] tensor (float, normalized)

    Losses:
      - Dynamics loss: MSE(predicted_z_next, actual_z_next)
      - Reward loss:   MSE(predicted_reward, actual_cos_angle)
    """
    optimizer = torch.optim.AdamW(world_model.get_trainable_params(), lr=lr, weight_decay=1e-4)
    dynamics_criterion = nn.MSELoss()
    reward_criterion = nn.MSELoss()

    N = data['obs'].shape[0]
    n_batches = N // batch_size

    print(f"\nTraining adapter on {N} frames, {epochs} epochs, batch_size={batch_size}")
    print(f"  Trainable params: {sum(p.numel() for p in world_model.get_trainable_params()):,}")

    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_dyn_loss = 0.0
        total_rew_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            obs_batch = data['obs'][idx].to(device)         # [B, T, H, W, C]
            next_obs_batch = data['next_obs'][idx].to(device)
            action_batch = data['action'][idx]               # [B, 1]
            reward_batch = data['reward'][idx].to(device)    # [B]

            # Convert continuous action → discrete index
            action_idx = torch.tensor(
                [continuous_action_to_idx(a) for a in action_batch],
                dtype=torch.long, device=device
            )

            # Encode current and next observations
            z = world_model.encode(obs_batch)          # [B, D]
            with torch.no_grad():
                z_next_actual = world_model.encode(next_obs_batch)  # [B, D]

            # Predict next latent
            z_next_pred = world_model.predict(z, action_idx)  # [B, D]

            # Predict reward from current latent
            rew_pred = world_model.get_reward(z).squeeze(-1)  # [B]

            # Losses
            dyn_loss = dynamics_criterion(z_next_pred, z_next_actual)
            rew_loss = reward_criterion(rew_pred, reward_batch.float())
            loss = dyn_loss + 0.5 * rew_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_dyn_loss += dyn_loss.item()
            total_rew_loss += rew_loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: dyn_loss={total_dyn_loss/n_batches:.4f}  "
                  f"rew_loss={total_rew_loss/n_batches:.4f}")

    print(f"  Final: dyn_loss={total_dyn_loss/n_batches:.4f}  "
          f"rew_loss={total_rew_loss/n_batches:.4f}")


def prepare_data(env, num_episodes=10, device='cuda'):
    """Collect data and optionally compute reward labels."""
    print(f"\nCollecting {num_episodes} episodes of random data...")
    data = collect_random_data(env, num_episodes=num_episodes)

    # Reward: use dm_control's built-in reward as proxy for pole uprightness
    # CartPole swingup reward is based on pole angle and cart position
    # Normalize to [-1, 1]
    rewards = data['reward']
    if rewards.std() > 0:
        data['reward'] = 2.0 * (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8) - 1.0

    print(f"  Collected {data['obs'].shape[0]} transitions")
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['vjepa', 'hamjepa'])
    parser.add_argument('--vjepa_ckpt', type=str, default=None,
                        help='Path to V-JEPA checkpoint (.pth.tar)')
    parser.add_argument('--hamjepa_cfg', type=str, default=None,
                        help='Path to HamJEPA training config (.yaml)')
    parser.add_argument('--hamjepa_ckpt', type=str, default=None,
                        help='Path to HamJEPA checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='./adapter_weights.pt',
                        help='Where to save adapter weights')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create world model
    if args.model == 'vjepa':
        if args.vjepa_ckpt is None:
            raise ValueError("--vjepa_ckpt required for V-JEPA")
        world_model = VJEPAWorldModel(args.vjepa_ckpt, device=device)
    else:
        if args.hamjepa_cfg is None or args.hamjepa_ckpt is None:
            raise ValueError("--hamjepa_cfg and --hamjepa_ckpt required for HamJEPA")
        world_model = HamJEPAWorldModel(args.hamjepa_cfg, args.hamjepa_ckpt, device=device)

    world_model = world_model.to(device)

    # Collect data & train
    env = CartPolePixelEnv(gravity=1.0, stack_frames=16 if args.model == 'vjepa' else 1)
    data = prepare_data(env, num_episodes=10)
    train_adapter(world_model, data, epochs=200, device=device)

    # Save adapter weights
    torch.save({
        'action_offsets': world_model.action_offsets.data,
        'reward_predictor': world_model.reward_predictor.state_dict(),
    }, args.output)
    print(f"\nAdapter saved to: {args.output}")


if __name__ == '__main__':
    main()
