"""
Physics-Aware Masking (PA-Masking) — Simple Heuristic Version

Improvements over IA-JEPA's pixel-diff²-only approach:
  1. Multi-scale motion: diff¹ (velocity) + diff² (acceleration)
  2. True spatiotemporal: per-tubelet importance, not spatial × temporal scalar
  3. Local contrast normalization: importance relative to neighbors, not absolute
  4. Temporal smoothing: reduce flickering masks across adjacent frames
  5. Region growing: mask coherent regions, not scattered individual patches

All operations are pure torch/numpy — zero learned parameters, ~2ms overhead/batch.

Reference:
  - IA-JEPA: masks high pixel-acceleration regions (Santoshpaidi et al.)
  - V-JEPA multiblock3d: random 3D block masking (Meta FAIR)
"""

from multiprocessing import Value
from logging import getLogger

import torch
import torch.nn.functional as F

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):
    """
    Physics-aware mask collator — same interface as multiblock3d.MaskCollator.

    Config keys per mask strategy (in cfgs_mask[i]):
        ratio:          float, fraction of patches to mask (default 0.6)
        physics_frac:   float, fraction of masked patches sampled by importance
                        vs uniformly (default 0.85, 15% random for exploration)
        temp_smooth:    float, temporal smoothing window in tubelet units (default 1)
        region_grow:    int, iterations of region growing on importance map (default 2)
        contrast_kernel: int, local contrast normalization kernel in patch units (default 3)
    """

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super().__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2

        self.mask_generators = []
        for m in cfgs_mask:
            mg = _PhysicsMaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                mask_ratio=m.get('ratio', 0.6),
                physics_frac=m.get('physics_frac', 0.85),
                temp_smooth=m.get('temp_smooth', 1),
                region_grow=m.get('region_grow', 2),
                contrast_kernel=m.get('contrast_kernel', 3),
            )
            self.mask_generators.append(mg)

    def step(self):
        for mg in self.mask_generators:
            mg.step()

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for mg in self.mask_generators:
            masks_enc, masks_pred = mg(batch_size=len(batch), videos=collated_batch)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _PhysicsMaskGenerator:
    def __init__(
        self,
        crop_size,
        num_frames,
        spatial_patch_size,
        temporal_patch_size,
        mask_ratio,
        physics_frac,
        temp_smooth,
        region_grow,
        contrast_kernel,
    ):
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2

        self.height = crop_size[0] // spatial_patch_size     # e.g. 14
        self.width = crop_size[1] // spatial_patch_size      # e.g. 14
        self.duration = num_frames // temporal_patch_size    # e.g. 8
        self.num_patches = self.duration * self.height * self.width  # e.g. 1568

        self.spatial_patch_size = spatial_patch_size[0]  # 16
        self.temporal_patch_size = temporal_patch_size    # 2
        self.num_frames = num_frames                      # 16

        self.mask_ratio = mask_ratio
        self.physics_frac = physics_frac
        self.temp_smooth = temp_smooth
        self.region_grow = region_grow
        self.contrast_kernel = contrast_kernel

        self.num_target = int(self.num_patches * mask_ratio)

        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    # ------------------------------------------------------------------
    # Improvement 1: Multi-scale motion features (velocity + acceleration)
    # ------------------------------------------------------------------
    def _extract_motion(self, videos):
        """
        videos: [B, C, T, H, W]

        Returns:
            velocity:    [B, T, Hp, Wp]  — first-order temporal diff per patch
            acceleration: [B, T, Hp, Wp] — second-order temporal diff per patch

        IA-JEPA uses only acceleration. We add velocity because:
          - A ball rolling at constant speed: accel ≈ 0, velocity > 0
          - IA-JEPA would miss this → the ball region wouldn't be masked
          - Velocity captures sustained motion, acceleration captures sudden changes
        """
        B, C, T, H, W = videos.shape

        # First-order: pixel motion
        diff1 = torch.abs(videos[:, :, 1:] - videos[:, :, :-1])           # [B,C,T-1,H,W]
        # Second-order: pixel acceleration
        diff2 = torch.abs(diff1[:, :, 1:] - diff1[:, :, :-1])             # [B,C,T-2,H,W]

        # Average over channels
        vel = diff1.mean(dim=1)    # [B, T-1, H, W]
        acc = diff2.mean(dim=1)    # [B, T-2, H, W]

        # Pad to full T frames (edges: replicate nearest valid value)
        vel = F.pad(vel, (0,0,0,0, 0,1), mode='replicate')  # [B, T, H, W]
        acc = F.pad(acc, (0,0,0,0, 1,1), mode='replicate')  # [B, T, H, W]

        # Spatial pooling to patch grid
        def pool_to_patches(x):
            """x: [B, T, H, W] → [B, T, Hp, Wp]"""
            return x.reshape(B, T, self.height, self.spatial_patch_size,
                             self.width, self.spatial_patch_size).mean(dim=(3, 5))

        vel_p = pool_to_patches(vel)  # [B, T=16, 14, 14]
        acc_p = pool_to_patches(acc)  # [B, T=16, 14, 14]

        # Temporal pooling to tubelet grid
        def pool_to_tubelets(x):
            """x: [B, T=16, Hp=14, Wp=14] → [B, Tdown=8, Hp=14, Wp=14]"""
            return x.reshape(B, self.duration, self.temporal_patch_size,
                             self.height, self.width).mean(dim=2)

        vel_t = pool_to_tubelets(vel_p)  # [B, 8, 14, 14]
        acc_t = pool_to_tubelets(acc_p)  # [B, 8, 14, 14]

        return vel_t, acc_t

    # ------------------------------------------------------------------
    # Improvement 2: True spatiotemporal importance
    # ------------------------------------------------------------------
    def _raw_importance(self, vel, acc):
        """
        Combine velocity and acceleration into a single per-tubelet importance.

        vel, acc: [B, Tdown, Hp, Wp]

        IA-JEPA: importance_spatial[h,w] × importance_temporal[t]
                 → loses spatiotemporal interaction

        Ours: importance[t,h,w] = α·vel[t,h,w] + (1-α)·acc[t,h,w]
              → each tubelet has independent score capturing WHEN and WHERE
              motion happens

        α balances sustained motion vs sudden changes (default: equal weight).
        """
        # Normalize each modality to [0, 1] per video (handles brightness variation)
        def per_video_norm(x):
            """Min-max normalize per video, per modality."""
            B = x.shape[0]
            xf = x.reshape(B, -1)
            vmin = xf.min(dim=1, keepdim=True).values.reshape(B, 1, 1, 1)
            vmax = xf.max(dim=1, keepdim=True).values.reshape(B, 1, 1, 1)
            denom = (vmax - vmin).clamp(min=1e-8)
            return (x - vmin) / denom

        vel_n = per_video_norm(vel)
        acc_n = per_video_norm(acc)

        # Equal weight: both velocity and acceleration matter
        importance = 0.5 * vel_n + 0.5 * acc_n  # [B, Tdown, Hp, Wp]
        return importance

    # ------------------------------------------------------------------
    # Improvement 3: Local contrast normalization
    # ------------------------------------------------------------------
    def _local_contrast(self, importance):
        """
        Normalize importance relative to LOCAL neighbors, not globally.

        Why: a patch in a uniformly moving scene (e.g., panning camera) has
        high absolute motion but is NOT physically interesting. Local contrast
        highlights patches that move MORE than their surroundings.

        importance: [B, Tdown, Hp, Wp]
        Returns:    [B, Tdown, Hp, Wp]  (higher = more salient relative to neighbors)
        """
        k = self.contrast_kernel  # default 3
        pad = k // 2

        # Spatial local mean via avg_pool
        B, T, H, W = importance.shape
        flat = importance.reshape(B * T, 1, H, W)

        local_mean = F.avg_pool2d(flat, kernel_size=k, stride=1, padding=pad)  # [B*T, 1, H, W]
        local_mean = local_mean.reshape(B, T, H, W)

        # Contrast = importance - local_mean (positive = more motion than neighbors)
        contrast = importance - local_mean

        # ReLU: we only care about patches MORE active than surroundings
        contrast = torch.relu(contrast)

        # Re-normalize per video
        B_flat = contrast.reshape(B, -1)
        denom = B_flat.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        contrast = contrast / denom.reshape(B, 1, 1, 1)

        return contrast

    # ------------------------------------------------------------------
    # Improvement 4: Temporal smoothing
    # ------------------------------------------------------------------
    def _temporal_smooth(self, importance):
        """
        Smooth importance along time to reduce flickering masks.

        importance: [B, Tdown, Hp, Wp]
        Returns:    [B, Tdown, Hp, Wp]

        Uses a simple 1D avg_pool along the temporal dimension.
        """
        if self.temp_smooth <= 1:
            return importance

        B, T, H, W = importance.shape
        # Permute to [B*H*W, 1, T] for 1D temporal pooling
        flat = importance.permute(0, 2, 3, 1).reshape(-1, 1, T)
        smoothed = F.avg_pool1d(flat, kernel_size=self.temp_smooth,
                                stride=1, padding=self.temp_smooth // 2)
        return smoothed.reshape(B, H, W, T).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    # Improvement 5: Region growing (coherent object-level masks)
    # ------------------------------------------------------------------
    def _region_grow(self, importance):
        """
        Expand high-importance regions to cover entire interacting objects.

        importance: [B, Tdown, Hp, Wp]
        Returns:    [B, Tdown, Hp, Wp]

        Algorithm: iterative spatial max-pool (dilation) on high-importance regions.
        This makes masked regions cover whole objects rather than just edges.
        """
        if self.region_grow <= 0:
            return importance

        B, T, H, W = importance.shape
        flat = importance.reshape(B * T, 1, H, W)

        # Threshold: top 30% patches are "physics seeds"
        threshold = importance.reshape(B, -1).quantile(0.7, dim=1)  # [B]
        threshold = threshold.reshape(B, 1, 1, 1)
        seeds = (importance > threshold).float()  # [B, T, H, W]
        seeds_flat = seeds.reshape(B * T, 1, H, W)

        # Dilate seeds via max_pool → expands to neighboring patches
        for _ in range(self.region_grow):
            seeds_flat = F.max_pool2d(seeds_flat, kernel_size=3, stride=1, padding=1)

        seeds = seeds_flat.reshape(B, T, H, W)

        # Blend: dilated seeds boost importance of nearby regions
        return importance * (1.0 + 0.5 * seeds)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def __call__(self, batch_size, videos):
        """
        videos: [B, C, T, H, W]  (already default_collate'd)

        Returns:
            context_idx: [B, K_context]  — flat indices of visible patches
            target_idx:  [B, K_target]   — flat indices of masked patches
        """
        B = videos.shape[0]
        N = self.num_patches
        K_target = self.num_target
        K_context = N - K_target

        # ── Step 1: Extract multi-scale motion ──
        vel, acc = self._extract_motion(videos)  # [B, Td, Hp, Wp] each

        # ── Step 2: Spatiotemporal importance ──
        importance = self._raw_importance(vel, acc)

        # ── Step 3: Local contrast normalization ──
        importance = self._local_contrast(importance)

        # ── Step 4: Temporal smoothing ──
        importance = self._temporal_smooth(importance)

        # ── Step 5: Region growing ──
        importance = self._region_grow(importance)

        # ── Step 6: Flatten and convert to sampling probabilities ──
        importance_flat = importance.reshape(B, N)  # [B, 1568]

        # Temperature-scaled softmax (temperature controls sharpness)
        temperature = 0.5  # lower = sharper (more concentrated on top patches)
        prob = torch.softmax(importance_flat / temperature, dim=1)  # [B, N]

        # ── Step 7: Sample target patches ──
        # physics_frac: sampled by importance (physics-guided)
        # 1-physics_frac: sampled uniformly (exploration, prevents mode collapse)
        n_physics = int(K_target * self.physics_frac)
        n_uniform = K_target - n_physics

        target_idx_list = []
        for b in range(B):
            if n_physics > 0:
                physics_idx = torch.multinomial(prob[b], n_physics, replacement=False)
            else:
                physics_idx = torch.empty(0, dtype=torch.long, device=videos.device)

            if n_uniform > 0:
                mask = torch.ones(N, dtype=torch.bool, device=videos.device)
                mask[physics_idx] = False
                remaining = torch.arange(N, device=videos.device)[mask]
                perm = torch.randperm(len(remaining), device=videos.device)[:n_uniform]
                uniform_idx = remaining[perm]
            else:
                uniform_idx = torch.empty(0, dtype=torch.long, device=videos.device)

            target_idx_list.append(torch.cat([physics_idx, uniform_idx]).long())

        target_idx = torch.stack(target_idx_list)  # [B, K_target]

        # ── Step 8: Context = all other patches ──
        all_idx = torch.arange(N, device=videos.device).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, N, dtype=torch.bool, device=videos.device)
        mask.scatter_(1, target_idx, False)
        context_idx = all_idx[mask].reshape(B, K_context)  # [B, K_context]

        return context_idx, target_idx
