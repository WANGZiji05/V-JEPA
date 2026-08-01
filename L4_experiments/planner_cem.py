"""
CEM (Cross-Entropy Method) Planner in Latent Space.

Uses a frozen world model (encoder + predictor + action adapter) to:
  1. Sample K action sequences of length H
  2. Roll out each sequence using the world model in latent space
  3. Score trajectories based on predicted reward
  4. Select top-K elite sequences, refit Gaussian distribution
  5. Repeat for N iterations → return first action of best sequence

Key design choices for CartPole-Swingup:
  - Discretized actions (5 bins) for efficient search
  - Latent-space rollout (no pixel decoding)
  - Reward = cos(pole_angle) predicted from latent by reward_predictor
"""

import torch
import torch.nn.functional as F


class CEMPlanner:
    """
    Cross-Entropy Method planner operating in latent space.

    Args:
        world_model: VJEPAWorldModel or HamJEPAWorldModel
        horizon: planning horizon (number of steps to look ahead)
        population: number of action sequences sampled per iteration (K)
        elite_frac: fraction of top sequences kept per iteration
        num_iter: CEM iterations
        action_dim: number of discrete actions (default 5)
        device: torch device
    """

    def __init__(
        self,
        world_model,
        horizon=5,
        population=100,
        elite_frac=0.1,
        num_iter=3,
        device='cuda',
    ):
        self.model = world_model
        self.H = horizon
        self.K = population
        self.elite_frac = elite_frac
        self.num_iter = num_iter
        self.action_dim = world_model.num_discrete_actions
        self.device = device

        self.n_elite = max(1, int(population * elite_frac))

    def plan(self, z_current):
        """
        Plan optimal action sequence from current latent state.

        z_current: [D] latent state vector

        Returns:
            best_action: int (discrete action index 0..action_dim-1)
            best_score:  float (predicted cumulative reward)
        """
        # Initialize action distribution: uniform over discrete actions
        logits = torch.zeros(self.H, self.action_dim, device=self.device)

        best_action_seq = None
        best_score = -float('inf')

        for iteration in range(self.num_iter):
            # Sample K action sequences from current distribution
            probs = F.softmax(logits / max(0.5, 1.0 - iteration * 0.2), dim=-1)
            # Gumbel-softmax sampling
            noise = -torch.log(-torch.log(torch.rand(self.K, self.H, self.action_dim,
                                                      device=self.device).clamp(min=1e-8)))
            sample_logits = torch.log(probs.unsqueeze(0).clamp(min=1e-8)) + noise
            actions = sample_logits.argmax(dim=-1)  # [K, H]

            # Roll out each action sequence
            scores = self._evaluate_sequences(z_current, actions)  # [K]

            # Update best
            top_score, top_idx = scores.max(dim=0)
            if top_score > best_score:
                best_score = top_score.item()
                best_action_seq = actions[top_idx]

            # Select elite sequences
            _, elite_idx = scores.topk(self.n_elite, dim=0)
            elite_actions = actions[elite_idx]  # [n_elite, H]

            # Refit: count actions at each horizon step
            logits = torch.zeros_like(logits)
            for t in range(self.H):
                counts = torch.bincount(elite_actions[:, t],
                                        minlength=self.action_dim).float()
                logits[t] = torch.log(counts + 1.0)  # Laplace smoothing

        # Return first action of best sequence
        if best_action_seq is None:
            return 2, best_score  # fallback: action=0 (no force)
        return best_action_seq[0].item(), best_score

    def _evaluate_sequences(self, z0, action_sequences):
        """
        Roll out K action sequences from z0 and compute cumulative reward.

        z0:               [D]
        action_sequences: [K, H] int tensor

        Returns: [K] cumulative reward scores
        """
        K, H = action_sequences.shape
        z = z0.unsqueeze(0).expand(K, -1)  # [K, D]

        total_reward = torch.zeros(K, device=self.device)

        for t in range(H):
            actions_t = action_sequences[:, t]  # [K]
            z = self.model.predict(z, actions_t)  # [K, D]
            reward = self.model.get_reward(z).squeeze(-1)  # [K]
            total_reward += reward

        return total_reward


class RandomPlanner:
    """Baseline: random action selection (no planning)."""

    def __init__(self, action_dim=5):
        self.action_dim = action_dim

    def plan(self, z_current):
        import random
        return random.randint(0, self.action_dim - 1), 0.0
