# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
V-JEPA Physion++ 注意力探针评估 —— 工具模块

本模块提供 Physion++ 数据集评估所需的工具函数：
- PhysionNormalize: Physion++ 专用的归一化参数
- make_physion_transforms: 构建适用于 Physion++ 的视频变换
"""

import torch
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.datasets.utils.video.randerase import RandomErasing


# Physion++ 数据集归一化参数
# 注意：这些是占位值（使用 ImageNet 标准统计量）
# 如果你的 Physion++ 渲染视频颜色分布有明显不同，
# 建议计算实际数据集的均值/标准差并替换这里的值
PHYSION_MEAN = (0.485, 0.456, 0.406)
PHYSION_STD = (0.229, 0.224, 0.225)


def make_physion_transforms(
    training=True,
    crop_size=224,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    num_views_per_clip=1,
    normalize=(PHYSION_MEAN, PHYSION_STD),
):
    """
    为 Physion++ 数据集创建视频变换。

    与 video_classification_frozen/utils.py 中的 make_transforms()
    保持相同的设计模式，但默认参数根据 Physion++ 数据集做了调整。

    Physion++ 视频特点：
    - 由 TDW 仿真引擎渲染，场景相对"干净"（无真实世界的复杂纹理）
    - 物体运动是核心信息，过度增强可能破坏物理运动线索
    - 因此不建议使用太强的数据增强（auto_augment 默认 False）

    参数:
        training (bool): 训练模式 / 验证模式
        crop_size (int): 最终裁剪尺寸（默认 224）
        random_horizontal_flip (bool): 是否随机水平翻转
            Physion++ 中物体运动方向可能很重要，建议设为 False
        random_resize_aspect_ratio (tuple): 随机缩放的宽高比范围
        random_resize_scale (tuple): 随机缩放的尺度范围
        reprob (float): 随机擦除概率
        auto_augment (bool): 是否使用 RandAugment 自动增强
        num_views_per_clip (int): 验证时的空间视角数
        normalize (tuple): (mean, std) 归一化参数

    返回:
        VideoTransform 或 EvalVideoTransform
    """

    if not training and num_views_per_clip > 1:
        # 多视角验证模式
        return EvalPhysionTransform(
            num_views_per_clip=num_views_per_clip,
            short_side_size=crop_size,
            normalize=normalize,
        )
    else:
        # 训练模式 / 单视角验证
        return PhysionTransform(
            training=training,
            random_horizontal_flip=random_horizontal_flip,
            random_resize_aspect_ratio=random_resize_aspect_ratio,
            random_resize_scale=random_resize_scale,
            reprob=reprob,
            auto_augment=auto_augment,
            crop_size=crop_size,
            normalize=normalize,
        )


class PhysionTransform(object):
    """
    Physion++ 视频变换（训练用）。

    变换流水线（训练模式）：
      1. numpy → PIL Image（如需 auto_augment）
      2. RandAugment 自动增强（如启用）
      3. PIL → Tensor [T, C, H, W]
      4. 重排为 [T, H, W, C] → 归一化 → 重排为 [C, T, H, W]
      5. 随机裁剪缩放
      6. 随机水平翻转（可选）
      7. 随机擦除（可选）

    变换流水线（验证模式）：
      1. Resize 短边
      2. 中心裁剪
      3. 转 Tensor + 归一化
    """

    def __init__(
        self,
        training=True,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        crop_size=224,
        normalize=(PHYSION_MEAN, PHYSION_STD),
    ):
        self.training = training

        # 验证时的标准变换
        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose([
            video_transforms.Resize(short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
        ])

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.crop_size = crop_size
        self.normalize = torch.tensor(normalize)

        # RandAugment 自动增强
        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        # 空间变换
        self.spatial_transform = video_transforms.random_resized_crop

        # 随机擦除
        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):
        if not self.training:
            return [self.eval_transform(buffer)]

        # 训练模式变换流水线
        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        if self.auto_augment:
            buffer = self.autoaug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # [T, C, H, W]
        buffer = buffer.permute(0, 2, 3, 1)  # [T, H, W, C]

        buffer = (buffer - self.normalize[0]) / self.normalize[1]
        buffer = buffer.permute(3, 0, 1, 2)  # [C, T, H, W]

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )

        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return [buffer]


class EvalPhysionTransform(object):
    """
    Physion++ 多视角验证变换。

    从缩放后的视频帧中裁剪多个空间视角。
    """

    def __init__(
        self,
        num_views_per_clip=1,
        short_side_size=224,
        normalize=(PHYSION_MEAN, PHYSION_STD),
    ):
        self.views_per_clip = num_views_per_clip
        self.short_side_size = short_side_size

        self.spatial_resize = video_transforms.Resize(
            short_side_size, interpolation='bilinear'
        )
        self.to_tensor = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
        ])

    def __call__(self, buffer):
        import numpy as np
        buffer = np.array(self.spatial_resize(buffer))
        T, H, W, C = buffer.shape

        num_views = self.views_per_clip
        side_len = self.short_side_size
        spatial_step = (max(H, W) - side_len) // (num_views - 1) if num_views > 1 else 0

        all_views = []
        for i in range(num_views):
            start = i * spatial_step
            if H > W:
                view = buffer[:, start:start + side_len, :, :]
            else:
                view = buffer[:, :, start:start + side_len, :]
            view = self.to_tensor(view)
            all_views.append(view)

        return all_views
