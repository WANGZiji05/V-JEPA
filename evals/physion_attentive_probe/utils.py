# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
V-JEPA Physion++ 注意力探针评估 —— 工具模块

本模块提供 Physion++ 数据集评估所需的视频变换。
所有变换基于 numpy + torch，不依赖 PIL/torchvision，
避免在某些 HPC 环境中因缺少 PIL 而无法运行。
"""

import numpy as np
import torch

from src.datasets.utils.video.volume_transforms import ClipToTensor


# Physion++ 数据集归一化参数（ImageNet 标准值）
PHYSION_MEAN = (0.485, 0.456, 0.406)
PHYSION_STD = (0.229, 0.224, 0.225)


def make_physion_transforms(
    training=True,
    crop_size=224,
    normalize=(PHYSION_MEAN, PHYSION_STD),
    **kwargs,  # 忽略其他不支持的增强参数
):
    """
    为 Physion++ 数据集创建视频变换。

    仅支持 eval 模式的基本变换（resize + centercrop + normalize）,
    不依赖 PIL。训练增强参数被忽略。
    """
    return PhysionEvalTransform(crop_size=crop_size, normalize=normalize)


class PhysionEvalTransform(object):
    """
    纯 numpy 实现的视频评估变换。
    流水线: Resize(短边) → CenterCrop → ToTensor → Normalize
    """

    def __init__(self, crop_size=224, normalize=(PHYSION_MEAN, PHYSION_STD)):
        self.crop_size = crop_size
        self.short_side = int(crop_size * 256 / 224)
        self.mean = np.array(normalize[0], dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.array(normalize[1], dtype=np.float32).reshape(1, 1, 1, 3)

    def __call__(self, buffer):
        """
        buffer: numpy array [T, H, W, C] (uint8 0-255) 或 list of PIL images
        返回: list of [tensor[C, T, H, W]] (归一化到 ImageNet 分布)
        """
        # 如果是 list (PIL images 或 numpy), 转为 numpy
        if isinstance(buffer, list):
            buffer = np.array([np.asarray(frame) for frame in buffer])

        # buffer: [T, H, W, C]
        T, H, W, C = buffer.shape

        # ---- Resize 短边 ----
        scale = self.short_side / min(H, W)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))

        # 简单的双线性插值 resize (对每帧)
        resized = np.zeros((T, new_h, new_w, C), dtype=buffer.dtype)
        for t in range(T):
            # 用简单的最近邻 + 平均做 resize（避免 scipy 依赖）
            h_ratio = H / new_h
            w_ratio = W / new_w
            for i in range(new_h):
                for j in range(new_w):
                    src_i = min(int(i * h_ratio), H - 1)
                    src_j = min(int(j * w_ratio), W - 1)
                    resized[t, i, j] = buffer[t, src_i, src_j]

        buffer = resized

        # ---- CenterCrop ----
        H_c, W_c = buffer.shape[1], buffer.shape[2]
        h_start = (H_c - self.crop_size) // 2
        w_start = (W_c - self.crop_size) // 2
        buffer = buffer[:, h_start:h_start + self.crop_size, w_start:w_start + self.crop_size, :]

        # ---- Normalize & convert to tensor ----
        buffer = buffer.astype(np.float32) / 255.0
        buffer = (buffer - self.mean) / self.std

        # [T, H, W, C] → [C, T, H, W]
        buffer = torch.from_numpy(buffer).permute(3, 0, 1, 2)

        return [buffer]
