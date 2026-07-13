# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# Random Tube Mask 生成器 —— 一种更简单的mask策略
# ============================================================================
# 与 MultiBlock3D Mask 不同，Random Tube Mask 使用一种更简单的策略：
# 在空间中随机遮罩一定比例的patch（如90%），但在时间维度上保持完整。
#
# 具体做法：
# 1. 在空间维度(H×W)随机选择一定比例的patch保留作为上下文
# 2. 被选中的patch在所有时间帧上都保留（形成"管道/tube"）
# 3. 其余的patch在所有时间帧上都被遮罩
#
# 这种mask策略的直觉：
# - 时间维度保持完整 → 预测器可以利用完整的时序信息
# - 空间随机遮罩 → 迫使模型理解空间语义
# - 比MultiBlock更简单 → 计算开销更小

from multiprocessing import Value
from logging import getLogger
import torch
import numpy as np

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):
    """
    随机管状mask收集器

    功能与multiblock3d的MaskCollator相同（管理多种mask策略），
    但内部使用更简单的随机空间遮罩。
    """

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                ratio=m.get('ratio'),  # 遮罩比例（如0.9表示遮罩90%的patch）
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        """步进随机种子"""
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        """为一个batch生成管状mask"""
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):
    """
    单个管状mask策略的生成器

    核心思想：
    1. 在空间维度随机选择 num_keep_spatial 个patch位置
    2. 这些位置在所有时间帧上都保留（形成管状结构）
    3. 其余位置在所有时间帧上都被遮罩
    """

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        ratio=0.9,  # 被遮罩的比例（例如0.9=90%的patch被遮罩）
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        # 转换到patch单位
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_patches_spatial = self.height * self.width  # 空间patch总数

        self.ratio = ratio

        # 要保留的（上下文）空间patch数量
        self.num_keep_spatial = int(self.num_patches_spatial * (1. - self.ratio))
        # 总的保留patch数 = 空间保留数 × 时间深度（管状结构）
        self.num_keep = self.num_keep_spatial * self.duration

        self._itr_counter = Value('i', -1)

    def step(self):
        """递增迭代计数器"""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch_size):
        """
        为一个batch生成管状mask

        流程：
        1. 在空间维度随机选择保留的patch位置
        2. 将所有时间帧的同一位置都设为保留（管状结构）
        3. 返回编码器mask（保留的patch索引）和预测器mask（遮罩的patch索引）
        """
        def sample_mask():
            """为单个样本生成mask"""
            # 在空间维度创建mask：0=遮罩, 1=保留
            mask = np.hstack([
                np.zeros(self.num_patches_spatial - self.num_keep_spatial),
                np.ones(self.num_keep_spatial),
            ])
            np.random.shuffle(mask)  # 随机打乱（随机选择保留位置）

            # 将所有时间帧的同一位置设为相同值（管状结构）
            mask = torch.tensor(np.tile(mask, (self.duration, 1)))
            mask = mask.flatten()

            # mask_p: 遮罩区域的索引（预测目标）
            mask_p = torch.argwhere(mask == 0).squeeze()
            # mask_e: 保留区域的索引（编码器上下文）
            mask_e = torch.nonzero(mask).squeeze()
            return mask_e, mask_p

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(batch_size):
            mask_e, mask_p = sample_mask()
            collated_masks_enc.append(mask_e)
            collated_masks_pred.append(mask_p)

        # 用default_collate堆叠成batch张量
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_masks_enc, collated_masks_pred
