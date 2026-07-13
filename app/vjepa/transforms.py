# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 视频数据增强 (Data Augmentation)
# ============================================================================
# 数据增强是自监督学习的关键组成部分。
# 通过对同一视频应用不同的随机变换，模型学习更鲁棒的特征。
#
# 主要增强策略（用于预训练）：
# 1. 随机缩放裁剪(RandomResizedCrop) —— 随机选一个区域然后缩放到固定尺寸
# 2. 水平翻转(RandomHorizontalFlip) —— 50%概率左右翻转
# 3. 运动偏移(Motion Shift) —— 首尾帧使用不同的裁剪区域（模拟相机运动）
# 4. 颜色归一化(Normalize) —— 减去均值除以标准差
#
# 数据增强流程：
#   原始视频帧 [T, H, W, C]
#   → Permute [C, T, H, W]
#   → 随机缩放裁剪
#   → 水平翻转
#   → 归一化
#   → [可选]随机擦除

import torch
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
from src.datasets.utils.video.randerase import RandomErasing


def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),    # 裁剪区域长宽比范围
    random_resize_scale=(0.3, 1.0),            # 裁剪区域相对于原图的大小比例
    reprob=0.0,                                 # 随机擦除概率
    auto_augment=False,                         # 是否使用自动增强（randaugment）
    motion_shift=False,                         # 是否使用运动偏移增强
    crop_size=224,                              # 最终裁剪尺寸
    normalize=((0.485, 0.456, 0.406),          # ImageNet标准均值
               (0.229, 0.224, 0.225))           # ImageNet标准标准差
):
    """创建数据增强pipeline（返回VideoTransform对象）"""
    _frames_augmentation = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=normalize,
    )
    return _frames_augmentation


class VideoTransform(object):
    """
    视频数据增强类

    对视频的每一帧应用相同的空间变换（保持帧之间的空间一致性）。
    如果启用了motion_shift，首尾帧会使用略微不同的裁剪区域。
    """

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):
        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size

        # 归一化参数
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)
        if not self.auto_augment:
            # 不使用auto_augment时，输入是uint8 [0,255]，需要scale到[0,1]
            self.mean *= 255.
            self.std *= 255.

        # 自动增强(RandAugment) —— 仅在评估时使用
        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        # 空间变换：随机裁剪（有/无运动偏移）
        self.spatial_transform = video_transforms.random_resized_crop_with_shift \
            if motion_shift else video_transforms.random_resized_crop

        # 随机擦除：随机遮罩掉图像中的一个矩形区域
        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):
        """
        对视频帧序列应用数据增强

        参数:
            buffer: 视频帧列表或张量

        返回:
            增强后的视频张量 [C, T, H, W]
        """
        if self.auto_augment:
            # 自动增强模式（评估时使用）：先转PIL，应用randaugment，再转tensor
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.autoaug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # [T, C, H, W]
            buffer = buffer.permute(0, 2, 3, 1)  # [T, H, W, C]
        else:
            buffer = torch.tensor(buffer, dtype=torch.float32)

        # 改变维度顺序：[T, H, W, C] → [C, T, H, W]
        buffer = buffer.permute(3, 0, 1, 2)

        # 应用空间变换（随机缩放裁剪）
        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )

        # 随机水平翻转（50%概率）
        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        # 颜色归一化
        buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)

        # 随机擦除
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)  # [C, T, H, W] → [T, C, H, W]
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # [T, C, H, W] → [C, T, H, W]

        return buffer


def tensor_normalize(tensor, mean, std):
    """
    张量归一化（非原地操作）

    将uint8 [0,255] 张量转换为归一化的float张量:
    output = (input/255 - mean) / std
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    张量归一化（原地操作，更节省显存）

    输入: [C, T, H, W]
    对每个通道C，执行: (value - mean[C]) / std[C]
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # [C, T*H*W] → [T*H*W, C]
    tensor.sub_(mean).div_(std)                 # 原地做 (x - mean) / std
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # 恢复 [C, T, H, W]
    return tensor
