# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 张量工具函数
# ============================================================================
# 包含：
# - trunc_normal_: 截断正态分布初始化（ViT权重初始化的标准方法）
# - repeat_interleave_batch: 按batch重复张量（用于将同一mask复制给多个clip）

import math
import torch
from logging import getLogger

logger = getLogger()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    截断正态分布初始化的底层实现（无梯度版本）

    截断正态分布：从正态分布 N(mean, std) 中采样，但只保留 [a, b] 范围内的值。
    这避免了极端权重值，使训练更稳定。

    实现方法：使用逆CDF变换
    1. 计算截断区间的CDF值 [l, u]
    2. 在 [2l-1, 2u-1] 的均匀分布中采样
    3. 用 erfinv (逆误差函数) 将均匀分布转换为标准正态分布
    4. 缩放和平移得到目标均值和标准差
    """
    def norm_cdf(x):
        """标准正态分布的累积分布函数(CDF)"""
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # 计算截断边界的CDF值
        l = norm_cdf((a - mean) / std)  # 下界对应的CDF
        u = norm_cdf((b - mean) / std)  # 上界对应的CDF

        # 在 [2l-1, 2u-1] 范围内均匀采样
        # (这是逆CDF变换的一个技巧，利用erfinv的特性)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # erfinv 将均匀分布 → 标准正态分布
        tensor.erfinv_()

        # 缩放到目标均值和标准差
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # 确保值在截断范围内
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    截断正态分布初始化（公开接口）

    默认在 [-2σ, +2σ] 范围内采样，大约覆盖95%的概率质量。

    在ViT中使用：
    截断正态分布初始化比标准正态分布初始化更稳定，
    因为极端值被截断了，不会产生异常大的初始激活值。
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x, B, repeat):
    """
    按batch重复张量 —— 用于将同一mask复制给同一视频的多个clip

    参数:
        x: 输入张量 [B_total, N, D]
        B: 原始batch大小
        repeat: 每个样本要复制多少次

    工作原理：
    将batch分成N=B_total/B组，每组内重复repeat次。
    例如：x有2个样本需要各自重复3次
    x = [a1, a2, b1, b2]
    B=2, repeat=3
    → [a1, a2, a1, a2, a1, a2, b1, b2, b1, b2, b1, b2]

    在V-JEPA中的用途：
    一个视频有num_clips个clip，每个clip共享相同的mask。
    但batch中每个clip都是独立条目，所以需要将mask复制num_clips次。
    """
    N = len(x) // B  # 每组有多少个样本
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x
