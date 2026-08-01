"""
Train DynamicsMLP + RewardPredictor on CartPole data.

Usage:
  python L4_experiments/train_adapter.py --model vjepa \
      --vjepa_ckpt /path/to/vitl16.pth.tar --output vjepa_dynamics.pt
  python L4_experiments/train_adapter.py --model hamjepa \
      --hamjepa_cfg ... --hamjepa_ckpt ... --output hamjepa_dynamics.pt
"""

import torch, argparse
import torch.nn as nn

from env_wrapper import CartPolePixelEnv, collect_random_data
from world_model import VJEPAWorldModel, HamJEPAWorldModel, continuous_action_to_idx


def train_dynamics(world_model, data, epochs=200, batch_size=256, lr=1e-3, device='cuda'):
    N = data['obs'].shape[0]

    # ── Pre-encode all frames ──
    print(f"\nPre-encoding {N} frames ...")
    z_all, z_next_all = [], []
    encode_bs = 2
    with torch.no_grad():
        for i in range(0, N, encode_bs):
            z_all.append(world_model.encode(data['obs'][i:i+encode_bs].to(device)).cpu())
            z_next_all.append(world_model.encode(data['next_obs'][i:i+encode_bs].to(device)).cpu())
            if (i // encode_bs + 1) % 200 == 0:
                print(f"  {min(i+encode_bs, N)}/{N}")
    z_all = torch.cat(z_all, dim=0)
    z_next_all = torch.cat(z_next_all, dim=0)
    rewards_all = data['reward']
    action_idx_all = torch.tensor(
        [continuous_action_to_idx(a) for a in data['action']], dtype=torch.long
    )
    print(f"  Done. z: {z_all.shape}")

    # ── Train DynamicsMLP + RewardPredictor ──
    optimizer = torch.optim.AdamW(world_model.get_trainable_params(), lr=lr, weight_decay=1e-4)
    dyn_criterion = nn.MSELoss()
    rew_criterion = nn.MSELoss()

    print(f"\nTraining: {epochs} epochs")
    print(f"  Params: {sum(p.numel() for p in world_model.get_trainable_params()):,}")

    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_dyn, total_rew, n = 0.0, 0.0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z = z_all[idx].to(device)
            z_next_gt = z_next_all[idx].to(device)
            a_idx = action_idx_all[idx].to(device)
            r_gt = rewards_all[idx].to(device)

            z_next_pred = world_model.dynamics(z, a_idx)
            r_pred = world_model.reward(z).squeeze(-1)

            dyn_loss = dyn_criterion(z_next_pred, z_next_gt)
            rew_loss = rew_criterion(r_pred, r_gt.float())
            loss = dyn_loss + 0.5 * rew_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_dyn += dyn_loss.item()
            total_rew += rew_loss.item()
            n += 1

        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: dyn_loss={total_dyn/n:.4f}  rew_loss={total_rew/n:.4f}")

    print(f"  Final: dyn_loss={total_dyn/n:.4f}  rew_loss={total_rew/n:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['vjepa', 'hamjepa'])
    parser.add_argument('--vjepa_ckpt', type=str, default=None)
    parser.add_argument('--hamjepa_cfg', type=str, default=None)
    parser.add_argument('--hamjepa_ckpt', type=str, default=None)
    parser.add_argument('--output', type=str, default='dynamics.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stack_frames', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.model == 'vjepa':
        world_model = VJEPAWorldModel(args.vjepa_ckpt, device=device)
    else:
        world_model = HamJEPAWorldModel(args.hamjepa_cfg, args.hamjepa_ckpt, device=device)
    world_model = world_model.to(device)

    # Collect data
    env = CartPolePixelEnv(gravity=1.0, stack_frames=args.stack_frames)
    data = collect_random_data(env, num_episodes=10)
    rewards = data['reward']
    if rewards.std() > 0:
        data['reward'] = 2.0 * (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8) - 1.0

    train_dynamics(world_model, data, device=device)

    torch.save({
        'dynamics': world_model.dynamics.state_dict(),
        'reward': world_model.reward.state_dict(),
    }, args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
