# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# Multi-Block 3D Mask 生成器 —— V-JEPA 的核心 mask 策略
# ============================================================================
# 这是 V-JEPA 最复杂也是最重要的组件之一。
#
# V-JEPA 的训练方式：
# 1. 给定一个视频，随机遮罩掉一些时空区域
# 2. 编码器(Encoder)只看到未遮罩的"上下文"(context)区域
# 3. 目标编码器(Target Encoder)看到完整视频
# 4. 预测器(Predictor)根据上下文特征预测被遮罩区域的特征
#
# Multi-Block Mask 的含义：
# 遮罩(mask)不是随机的单个像素，而是由多个3D时空块(block)组成。
# 每个block是一个 (时间深度 × 高度 × 宽度) 的立方体区域。
# 多个block散落在视频的不同位置，共同构成被遮罩的"目标"区域。
#
# 为什么不随机遮罩单个patch？
# 单个patch太容易从相邻patch推理出来（只需要复制），
# 遮罩大块区域迫使模型学习更高级的语义理解。
#
# 配置文件中可以定义多种mask策略（如2种）：
#   - 策略1: 8个小块（短时小空间），训练模型理解局部细节
#   - 策略2: 2个大块（长时大空间），训练模型理解全局语义
# 两种策略的结果会分别计算loss，模型需要同时做好两种预测。

import math
from multiprocessing import Value  # 多进程共享计数器（用于确保不同worker生成不同的mask）
from logging import getLogger
import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):
    """
    Mask 收集器 —— 管理多种mask策略，为每个batch生成mask

    工作流程：
    1. 初始化时根据配置创建多个_MaskGenerator（每个对应一种mask策略）
    2. 每次被调用时，对batch中的每个样本，用每种策略生成mask
    3. 返回：原始batch数据 + 编码器mask列表 + 预测器(target) mask列表
    """

    def __init__(
        self,
        cfgs_mask,           # mask配置列表，每个元素定义一种mask策略
        crop_size=(224, 224),  # 视频的空间尺寸
        num_frames=16,         # 视频帧数
        patch_size=(16, 16),   # 空间patch尺寸
        tubelet_size=2,        # 时间tubelet尺寸
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            # 为每种mask策略创建一个生成器
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                # 目标区域的空间尺寸比例范围（如0.15~0.15表示固定15%）
                spatial_pred_mask_scale=m.get('spatial_scale'),
                # 目标区域的时间尺寸比例范围（如1.0~1.0表示所有帧）
                temporal_pred_mask_scale=m.get('temporal_scale'),
                # block的长宽比范围
                aspect_ratio=m.get('aspect_ratio'),
                # 目标区域由几个block组成
                npred=m.get('num_blocks'),
                # 上下文区域最多跨越多少帧
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                # 上下文区域最多保留多少patch
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        """步进所有mask生成器的随机种子，确保每次迭代生成不同的mask"""
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        """
        为一个batch的视频生成mask

        返回:
            collated_batch: 原始视频数据
            collated_masks_enc: 编码器mask列表（每种策略一个mask）
               每个mask包含的是【保留的patch索引】（编码器能看到的部分）
            collated_masks_pred: 预测器mask列表（每种策略一个mask）
               每个mask包含的是【被遮罩的patch索引】（需要预测的部分）

        注意：编码器mask和预测器mask是互补的！
              mask_enc ∪ mask_pred = 所有patch（但可能有重叠）
              mask_enc ∩ mask_pred = 空集（编码器看不到目标区域）
        """
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            # 为整个batch生成这种策略下的mask
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):
    """
    单个mask策略的生成器（内部类）

    核心思想：
    1. 先确定目标block的尺寸（时间t、空间h×w）
    2. 在视频中随机放置npred个这样的block
    3. 被block覆盖的patch → 目标区域（预测器要预测的）
    4. 未被覆盖的patch → 上下文区域（编码器能看到的）

    目标block的尺寸由以下参数控制：
    - spatial_scale: 空间占多少比例（如0.15=空间覆盖15%）
    - temporal_scale: 时间占多少比例（如1.0=覆盖所有帧）
    - aspect_ratio: block的长宽比范围
    - npred: 放置多少个block（多个小block vs 少量大block）
    """

    def __init__(
        self,
        crop_size=(224, 224),       # 视频空间尺寸（像素）
        num_frames=16,               # 视频帧数
        spatial_patch_size=(16, 16), # 空间patch尺寸
        temporal_patch_size=2,       # 时间tubelet尺寸
        spatial_pred_mask_scale=(0.2, 0.8),  # 目标区域空间比例范围
        temporal_pred_mask_scale=(1.0, 1.0), # 目标区域时间比例范围
        aspect_ratio=(0.3, 3.0),     # block长宽比范围
        npred=1,                     # 目标区域由几个block组成
        max_context_frames_ratio=1.0,  # 上下文最多跨越多少比例帧
        max_keep=None,               # 上下文最多保留多少patch
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size

        # 转换为patch单位的尺寸
        # 例如：224/16 = 14 个空间patch，16/2 = 8 个时间tubelet
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred  # 目标block数量
        # 上下文区域的最大时间跨度
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep

        # 多进程共享的迭代计数器
        # Value('i', -1) 创建一个在fork后所有子进程共享的整数
        self._itr_counter = Value('i', -1)

    def step(self):
        """递增迭代计数器，返回当前值作为随机种子"""
        i = self._itr_counter
        with i.get_lock():  # 多进程安全的自增
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator,
        temporal_scale,     # 时间比例范围
        spatial_scale,      # 空间比例范围
        aspect_ratio_scale  # 长宽比范围
    ):
        """
        采样一个目标block的尺寸

        从配置的范围中随机采样block的：
        - 时间深度 t
        - 空间高度 h
        - 空间宽度 w

        采样的block必须具有正确的面积（由spatial_scale控制）和长宽比（由aspect_ratio控制）。
        """
        # 采样时间深度
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # 采样空间面积
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        # spatial_num_keep 是 block 的面积（patches单位）
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # 采样长宽比
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # 根据面积和长宽比计算高度和宽度
        # area = h * w,  aspect_ratio = h / w
        # → h = sqrt(area * ar),  w = sqrt(area / ar)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)  # 不能超出视频高度
        w = min(w, self.width)   # 不能超出视频宽度

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        """
        在视频中随机放置一个block并生成mask

        返回一个3D mask，其中：
        - 1 = 保留（上下文区域，编码器可以看到）
        - 0 = 遮罩（目标区域，需要预测）

        参数:
            b_size: (t, h, w) block的时空尺寸

        block的位置是随机采样的：
        - 时间起始: 随机在 [0, duration - t]
        - 高度起始: 随机在 [0, height - h]
        - 宽度起始: 随机在 [0, width - w]
        """
        t, h, w = b_size

        # 随机采样block的起始位置
        top = torch.randint(0, self.height - h + 1, (1,))    # 高度起始
        left = torch.randint(0, self.width - w + 1, (1,))    # 宽度起始
        start = torch.randint(0, self.duration - t + 1, (1,))  # 时间起始

        # 创建全1的mask（默认所有patch都是上下文）
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        # 将block区域设置为0（遮罩/目标区域）
        mask[start:start+t, top:top+h, left:left+w] = 0

        # 如果上下文有时间限制，只保留前max_context_duration帧
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        return mask

    def __call__(self, batch_size):
        """
        为一个batch生成mask

        流程：
        1. 用固定种子采样block尺寸（确保所有样本使用相同大小的block）
        2. 对每个样本，用随机位置放置npred个block
        3. 所有被block覆盖的区域 = 目标区域（需要预测）
        4. 未被覆盖的区域 = 上下文区域（编码器可见）

        返回:
            collated_masks_enc: [B, K_enc] 编码器mask（保留的patch索引）
            collated_masks_pred: [B, K_pred] 预测器mask（遮罩的patch索引）
        """
        # 步骤1：用固定种子采样block尺寸
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        # 步骤2：为batch中每个样本生成mask
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                # 初始化为全1（全部保留）
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                # 放置npred个block（每次乘法将block区域设为0）
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()  # 展平为1D

                # 找到遮罩区域的索引（目标区域）
                mask_p = torch.argwhere(mask_e == 0).squeeze()
                # 找到保留区域的索引（上下文区域）
                mask_e = torch.nonzero(mask_e).squeeze()

                # 确保上下文不为空
                empty_context = len(mask_e) == 0
                if not empty_context:
                    # 记录最小保留/遮罩数，后续会截断到统一长度
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        # 步骤3：截断到最小长度（确保batch中所有样本的mask长度一致）
        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
