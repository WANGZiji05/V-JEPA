# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# Patch Embedding —— 将原始图像/视频像素转换为特征向量(token)
# ============================================================================
# 这是ViT处理视觉信息的第一步：把连续的像素空间离散化为patch序列。
# 类比NLP：图像 = 一篇文章，patch = 一个单词，patch embedding = 词向量(word embedding)
#
# 2D Patch (用于图像): 用2D卷积将 [C, H, W] 转换为 [D, H/P, W/P]
# 3D Patch (用于视频): 用3D卷积将 [C, T, H, W] 转换为 [D, T/Tubelet, H/P, W/P]

import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    2D 图像 Patch Embedding

    将一张图像切分成不重叠的小方块(patch)，每个patch通过卷积映射成一个向量。

    例如：输入 [B, 3, 224, 224] 的图片
         使用 patch_size=16
         卷积核大小 = 16×16, 步长 = 16（不重叠）
         输出 [B, 768, 14, 14]  →  flatten →  [B, 196, 768]
         即：14×14=196个patch，每个patch是768维向量
    """
    def __init__(
        self,
        patch_size=16,      # patch的空间尺寸（正方形）
        in_chans=3,          # 输入通道数（RGB=3）
        embed_dim=768        # 输出特征维度（即token维度）
    ):
        super().__init__()
        self.patch_size = patch_size
        # 卷积操作 = 天然的patch切分 + 线性投影
        # kernel_size=patch_size, stride=patch_size 意味着patch之间不重叠
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        输入: [B, C, H, W]  例如 [8, 3, 224, 224]
        中间: [B, D, H/P, W/P]  例如 [8, 768, 14, 14]
        输出: [B, N, D]  例如 [8, 196, 768]
              其中 N = (H/P) * (W/P) = 14*14 = 196
        """
        B, C, H, W = x.shape
        # 卷积投影: [B, C, H, W] → [B, D, H/P, W/P]
        x = self.proj(x)
        # Flatten: [B, D, H/P, W/P] → [B, D, (H/P)*(W/P)]
        # Transpose: → [B, (H/P)*(W/P), D]  (把特征维放到最后，符合Transformer输入习惯)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    3D 视频 Patch (Tubelet) Embedding

    将视频切分成时空块(tubelet = 管状小块)，每个tubelet映射成一个向量。

    与2D patch的区别：
    - 2D patch只在空间维度(H,W)切分
    - 3D tubelet在时空维度(T,H,W)同时切分，每个tubelet包含tubelet_size帧的信息

    例如：输入 [B, 3, 16, 224, 224] 的视频
         使用 patch_size=16, tubelet_size=2
         卷积核大小 = (2, 16, 16), 步长 = (2, 16, 16)
         输出 [B, 768, 8, 14, 14]  →  flatten →  [B, 1568, 768]
         即：8×14×14=1568个tubelet token
    """
    def __init__(
        self,
        patch_size=16,         # 空间patch尺寸
        tubelet_size=2,        # 时间维度上每个tubelet包含的帧数
        in_chans=3,            # 输入通道数
        embed_dim=768,         # 输出特征维度
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # 3D卷积：同时在时间和空间维度上滑动
        # kernel_size 和 stride 都是 (tubelet_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        """
        输入: [B, C, T, H, W]  例如 [8, 3, 16, 224, 224]
        中间: [B, D, T/Tubelet, H/P, W/P]  例如 [8, 768, 8, 14, 14]
        输出: [B, N, D]  例如 [8, 1568, 768]
              其中 N = (T/Tubelet) * (H/P) * (W/P) = 8*14*14 = 1568
        """
        B, C, T, H, W = x.shape
        x = self.proj(x)  # 3D卷积投影
        # flatten(2): 将 T, H, W 三个空间维度展平
        x = x.flatten(2).transpose(1, 2)
        return x
