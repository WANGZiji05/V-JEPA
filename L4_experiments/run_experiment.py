"""
L4 Experiment: MPC + CEM with frozen V-JEPA / HamJEPA predictors as world models.

Evaluates whether HamJEPA's Hamiltonian-structured predictor provides
better world modeling for model-predictive control, especially under
physical domain shift (gravity mutation).

Experiment groups (100 episodes each):
  1. V-JEPA  + default gravity (1×)
  2. HamJEPA + default gravity (1×)
  3. V-JEPA  + mutant gravity (2×)
  4. HamJEPA + mutant gravity (2×)

Plus random-action baseline for sanity check.

Usage:
  # Step 1: Train adapters
  python L4_experiments/train_adapter.py --model vjepa \
      --vjepa_ckpt /path/to/vith16.pth.tar --output vjepa_adapter.pt
  python L4_experiments/train_adapter.py --model hamjepa \
      --hamjepa_cfg /path/to/physion_hjepa_mv.yaml \
      --hamjepa_ckpt /path/to/physion_hjepa_mv.pth --output hamjepa_adapter.pt

  # Step 2: Run experiments
  python L4_experiments/run_experiment.py --all
"""

import os, sys, time, argparse
import torch
import numpy as np

from env_wrapper import CartPolePixelEnv
from world_model import (
    VJEPAWorldModel, HamJEPAWorldModel,
    continuous_action_to_idx, idx_to_continuous_action,
)
from planner_cem import CEMPlanner, RandomPlanner


# ── CartPole success criteria ──
# Episode is "successful" if pole stays near upright (> cos(30°) ≈ 0.866)
# for at least 80% of episode steps (i.e., 160 out of 200 steps)
SUCCESS_COS_THRESHOLD = 0.866  # cos(30°)
SUCCESS_STEP_RATIO = 0.8


def run_episode(env, planner, world_model, max_steps=200, device='cuda'):
    """
    Run one episode using the CEM planner.

    Returns:
        success: bool
        total_reward: float
        upright_ratio: float (fraction of steps where pole is near upright)
        episode_length: int
    """
    obs = env.reset().to(device)
    total_reward = 0.0
    upright_steps = 0
    step = 0

    for step in range(max_steps):
        # Encode observation → latent state
        with torch.no_grad():
            z = world_model.encode(obs.unsqueeze(0)).squeeze(0)  # [D]

        # Plan action via CEM
        action_idx, _ = planner.plan(z)

        # Map back to continuous action
        action = idx_to_continuous_action(action_idx)

        # Step environment
        obs, reward, done, _ = env.step(np.array([action]))
        obs = obs.to(device)
        total_reward += reward

        # Check if pole is upright
        # Use dm_control reward (≈1.0 = upright, ≈0 = hanging)
        if reward > 0.85:
            upright_steps += 1

        if done:
            break

    episode_length = step + 1
    upright_ratio = upright_steps / max(episode_length, 1)
    success = upright_ratio >= SUCCESS_STEP_RATIO

    return success, total_reward, upright_ratio, episode_length


def run_experiment_group(world_model, gravity, num_episodes, label, device='cuda',
                         cem_horizon=5, cem_population=100, cem_iter=3):
    """
    Run a group of episodes with given gravity setting.
    """
    env = CartPolePixelEnv(gravity=gravity, stack_frames=16)
    planner = CEMPlanner(
        world_model, horizon=cem_horizon, population=cem_population,
        num_iter=cem_iter, device=device
    )

    successes = 0
    rewards = []
    upright_ratios = []
    lengths = []

    print(f"\n{'='*60}")
    print(f"Group: {label} (gravity={gravity}×)")
    print(f"{'='*60}")

    t0 = time.time()
    for ep in range(num_episodes):
        success, total_reward, upright_ratio, ep_len = run_episode(
            env, planner, world_model, device=device
        )
        successes += int(success)
        rewards.append(total_reward)
        upright_ratios.append(upright_ratio)
        lengths.append(ep_len)

        if (ep + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (ep + 1) / elapsed * 60
            print(f"  [{ep+1:3d}/{num_episodes}] success_rate={successes/(ep+1):.2%}  "
                  f"avg_reward={np.mean(rewards):.2f}  {rate:.1f} ep/min")

    elapsed = time.time() - t0
    success_rate = successes / num_episodes
    print(f"\n  Final: success_rate={success_rate:.2%}  "
          f"avg_reward={np.mean(rewards):.2f}  "
          f"avg_upright={np.mean(upright_ratios):.2%}  "
          f"time={elapsed:.0f}s")

    return {
        'label': label,
        'gravity': gravity,
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'avg_reward': np.mean(rewards),
        'avg_upright_ratio': np.mean(upright_ratios),
        'avg_episode_length': np.mean(lengths),
        'elapsed_sec': elapsed,
    }


def run_random_baseline(num_episodes=100, gravity=1.0, device='cuda'):
    """Random action baseline to verify CEM is actually planning."""
    env = CartPolePixelEnv(gravity=gravity, stack_frames=1)
    planner = RandomPlanner()

    successes = 0
    for ep in range(num_episodes):
        obs = env.reset()
        upright_steps = 0
        for step in range(200):
            action_idx, _ = planner.plan(None)
            action = idx_to_continuous_action(action_idx)
            obs, reward, done, _ = env.step(np.array([action]))
            if reward > 0.85:
                upright_steps += 1
            if done:
                break
        upright_ratio = upright_steps / min(step + 1, 200)
        successes += int(upright_ratio >= SUCCESS_STEP_RATIO)
        if (ep + 1) % 20 == 0:
            print(f"  [Random baseline] {ep+1}/{num_episodes}  "
                  f"success_rate={successes/(ep+1):.2%}")

    rate = successes / num_episodes
    print(f"\n  Random baseline success_rate: {rate:.2%}")
    return rate


def main():
    parser = argparse.ArgumentParser(description='L4: MPC with frozen predictors')
    parser.add_argument('--all', action='store_true', help='Run all 4 groups + baseline')

    # Model paths
    parser.add_argument('--vjepa_ckpt', type=str, default=None)
    parser.add_argument('--vjepa_adapter', type=str, default=None)
    parser.add_argument('--hamjepa_cfg', type=str, default=None)
    parser.add_argument('--hamjepa_ckpt', type=str, default=None)
    parser.add_argument('--hamjepa_adapter', type=str, default=None)

    # Experiment settings
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--cem_horizon', type=int, default=5)
    parser.add_argument('--cem_population', type=int, default=100)
    parser.add_argument('--cem_iter', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='L4_results.txt')

    args = parser.parse_args()

    if not args.all:
        print("Usage: python run_experiment.py --all [--vjepa_ckpt PATH] [...]")
        print("       See --help for all options")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load models ──
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)

    vjepa_model = VJEPAWorldModel(args.vjepa_ckpt, device=device).to(device)
    if args.vjepa_adapter:
        print(f"Loading V-JEPA dynamics from {args.vjepa_adapter}")
        ckpt = torch.load(args.vjepa_adapter, map_location=device)
        vjepa_model.dynamics.load_state_dict(ckpt['dynamics'])
        vjepa_model.reward.load_state_dict(ckpt['reward'])

    hamjepa_model = HamJEPAWorldModel(
        args.hamjepa_cfg, args.hamjepa_ckpt, device=device
    ).to(device)
    if args.hamjepa_adapter:
        print(f"Loading HamJEPA dynamics from {args.hamjepa_adapter}")
        ckpt = torch.load(args.hamjepa_adapter, map_location=device)
        hamjepa_model.dynamics.load_state_dict(ckpt['dynamics'])
        hamjepa_model.reward.load_state_dict(ckpt['reward'])

    # ── Resume support ──
    import json as _json
    results_path = args.output.replace('.txt', '.json')
    saved = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            saved = _json.load(f)
        print(f"Resume: {len(saved.get('groups', []))} groups already completed")

    all_results = saved.get('groups', [])
    completed_labels = {r['label'] for r in all_results}

    def save_progress():
        with open(results_path, 'w') as f:
            _json.dump({
                'random_baseline': random_rate,
                'groups': all_results,
            }, f, indent=2)

    # ── Random baseline (no world model) ──
    if 'random_baseline' in saved:
        random_rate = saved['random_baseline']
        print(f"\nRandom baseline (cached): {random_rate:.2%}")
    else:
        print("\n" + "="*60)
        print("Random Action Baseline")
        print("="*60)
        random_rate = run_random_baseline(num_episodes=args.episodes, device=device)
        save_progress()

    # ── Experiment groups ──
    groups = [
        (vjepa_model, 1.0, "V-JEPA (1× gravity)"),
        (hamjepa_model, 1.0, "HamJEPA (1× gravity)"),
        (vjepa_model, 2.0, "V-JEPA (2× gravity)"),
        (hamjepa_model, 2.0, "HamJEPA (2× gravity)"),
    ]

    for model, gravity, label in groups:
        if label in completed_labels:
            print(f"\nSkipping (already done): {label}")
            continue

        result = run_experiment_group(
            model, gravity=gravity, num_episodes=args.episodes,
            label=label, device=device,
            cem_horizon=args.cem_horizon, cem_population=args.cem_population,
            cem_iter=args.cem_iter,
        )
        all_results.append(result)
        save_progress()

    # ── Report ──
    lines = []
    lines.append("="*70)
    lines.append("L4 Experiment Results: MPC with Frozen Predictors")
    lines.append("="*70)
    lines.append(f"  Random baseline success rate: {random_rate:.2%}")
    lines.append("")
    lines.append(f"  {'Group':<25} {'Success':>8} {'AvgReward':>10} {'Upright%':>9}")
    lines.append(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*9}")

    vjepa_in = None
    vjepa_out = None
    hamjepa_in = None
    hamjepa_out = None

    for r in all_results:
        lines.append(
            f"  {r['label']:<25} {r['success_rate']:>7.2%} "
            f"{r['avg_reward']:>10.2f} {r['avg_upright_ratio']:>8.2%}"
        )
        if '1×' in r['label'] and 'V-JEPA' in r['label']:
            vjepa_in = r['success_rate']
        elif '2×' in r['label'] and 'V-JEPA' in r['label']:
            vjepa_out = r['success_rate']
        elif '1×' in r['label'] and 'HamJEPA' in r['label']:
            hamjepa_in = r['success_rate']
        elif '2×' in r['label'] and 'HamJEPA' in r['label']:
            hamjepa_out = r['success_rate']

    lines.append("")
    lines.append("  Gravity Robustness (lower decay = better):")
    if vjepa_in and vjepa_out:
        vjepa_decay = (vjepa_in - vjepa_out) / max(vjepa_in, 0.01)
        lines.append(f"    V-JEPA  decay: {vjepa_decay:+.2%}")
    if hamjepa_in and hamjepa_out:
        hamjepa_decay = (hamjepa_in - hamjepa_out) / max(hamjepa_in, 0.01)
        lines.append(f"    HamJEPA decay: {hamjepa_decay:+.2%}")
    lines.append("="*70)

    report = '\n'.join(lines)
    print('\n' + report)

    with open(args.output, 'w') as f:
        f.write(report + '\n')
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
