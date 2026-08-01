"""
CartPole-Swingup environment wrapper with pixel observations.

Returns 224×224 RGB frames suitable for V-JEPA/HamJEPA encoders.
Supports frame stacking (for V-JEPA's video input) and gravity mutation.
"""

from dm_control import suite
import numpy as np
import torch
import torch.nn.functional as F

# ImageNet normalization (standard for both V-JEPA and HamJEPA)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CartPolePixelEnv:
    """
    CartPole-Swingup with pixel observations (224×224 RGB).

    Args:
        gravity: gravity multiplier (1.0 = default 9.81 m/s², 2.0 = 19.62 m/s²)
        stack_frames: number of frames to stack (1 for single image, 16 for V-JEPA video)
        render_size: output frame size (default 224×224)
    """

    def __init__(self, gravity=1.0, stack_frames=1, render_size=224):
        self.env = suite.load('cartpole', 'swingup')
        self.stack_frames = stack_frames
        self.render_size = render_size
        self._set_gravity(gravity)
        self.frame_buffer = []  # for frame stacking

        # Action space: continuous [-1, 1] force on cart
        self.action_spec = self.env.action_spec()

    def _set_gravity(self, multiplier):
        """Override gravity in the physics model."""
        physics = self.env.physics
        # MuJoCo gravity vector: [0, 0, -g]
        default_g = 9.81
        physics.model.opt.gravity[2] = -default_g * multiplier

    def reset(self):
        """Reset env and return initial observation tensor."""
        time_step = self.env.reset()
        frame = self._render_frame()
        self.frame_buffer = [frame] * self.stack_frames
        return self._get_obs()

    def step(self, action):
        """Take action, return (obs, reward, done, info)."""
        # Clip action to valid range
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)
        if action.ndim > 0:
            action = action.flatten()

        time_step = self.env.step(action)
        frame = self._render_frame()
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.stack_frames:
            self.frame_buffer.pop(0)

        obs = self._get_obs()
        reward = time_step.reward or 0.0
        done = time_step.last()
        return obs, reward, done, {}

    def _render_frame(self):
        """Render a single 224×224 RGB frame."""
        # dm_control renders at 240×320 by default, resize to target
        img = self.env.physics.render(
            height=self.render_size, width=self.render_size, camera_id=0
        )
        return img  # [H, W, C] uint8

    def _get_obs(self):
        """Return observation: single image or frame stack."""
        frames = np.stack(self.frame_buffer, axis=0)  # [T, H, W, C]
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - _MEAN.reshape(1, 1, 1, 3)) / _STD.reshape(1, 1, 1, 3)
        return torch.from_numpy(frames)  # [T, H, W, C]

    @property
    def action_dim(self):
        return self.action_spec.shape[0]  # 1 (continuous force)


def collect_random_data(env, num_episodes=5, steps_per_episode=200):
    """
    Collect (obs, action, next_obs, reward) tuples using random policy.
    Used to train the action adapter and reward predictor.
    """
    data = {'obs': [], 'action': [], 'next_obs': [], 'reward': []}

    for ep in range(num_episodes):
        obs = env.reset()
        for _ in range(steps_per_episode):
            action = np.random.uniform(-1, 1, size=env.action_dim)
            next_obs, reward, done, _ = env.step(action)
            data['obs'].append(obs)
            data['action'].append(torch.tensor(action, dtype=torch.float32))
            data['next_obs'].append(next_obs)
            data['reward'].append(reward)
            obs = next_obs
            if done:
                break
        print(f"  Collected episode {ep+1}/{num_episodes}")

    # Stack into tensors
    data['obs']     = torch.stack(data['obs'])      # [N, T, H, W, C]
    data['next_obs'] = torch.stack(data['next_obs'])
    data['action']  = torch.stack(data['action'])    # [N, 1]
    data['reward']  = torch.tensor(data['reward'])
    return data
