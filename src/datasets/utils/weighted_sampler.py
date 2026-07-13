# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 加权采样器 —— 支持不平衡数据集的分布式采样
# ============================================================================
# V-JEPA 使用多个不同的视频数据集进行预训练（如 Kinetics、SSv2、HowTo100M），
# 这些数据集的大小差异很大。
# 加权采样器确保每个数据集被采样的总概率与配置的权重成正比，
# 防止小数据集被"淹没"在大数据集中。
#
# 核心类:
# - CustomWeightedRandomSampler: 支持超过2^24样本的加权采样
# - DistributedWeightedSampler: 分布式版本的加权采样器

from typing import Iterator, Optional
from operator import itemgetter
import numpy as np

import torch
from torch.utils.data import (
    Dataset,
    Sampler,
    DistributedSampler,
    WeightedRandomSampler
)


class DatasetFromSampler(Dataset):
    """
    从采样器创建数据集（包装器）

    这是一个技巧：将采样器的输出包装成Dataset，
    以便能被DistributedSamplerWrapper使用。
    """
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    分布式采样器包装器

    将任意PyTorch采样器转换为分布式版本。
    工作原理：
    1. 先用内部的sampler生成一个全局采样序列
    2. 然后用DistributedSampler的标准逻辑分割给各个rank
    """
    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        # itemgetter用于从采样结果中提取对应rank的样本
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    自定义加权随机采样器

    PyTorch原版的WeightedRandomSampler使用torch.multinomial，
    最多支持2^24个样本。这个版本使用numpy.random.choice，
    可以处理任意大小的数据集。

    在V-JEPA中使用的场景：
    VideoMix2M数据集可能包含数千万个视频，
    远超PyTorch原版的限制。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        # 使用numpy的choice（支持任意大小）
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement  # 是否允许重复采样
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class DistributedWeightedSampler(DistributedSamplerWrapper):
    """
    分布式加权采样器 —— V-JEPA 处理多数据集的核心

    使用方法：
    1. 计算每个样本的权重 = 数据集权重 / 该数据集样本数
    2. 用CustomWeightedRandomSampler做全局加权采样
    3. 用DistributedSamplerWrapper分配到各个GPU

    例子：
    数据集A: 10000个样本, 权重=1.0 → 每个样本权重=0.0001
    数据集B: 1000个样本, 权重=0.5  → 每个样本权重=0.0005
    结果：数据集B的样本被采样到的概率是数据集A的5倍（尽管B更小）
    """
    def __init__(
        self,
        weights,                              # 每个样本的权重
        num_replicas: Optional[int] = None,   # GPU数量
        rank: Optional[int] = None,           # 当前GPU编号
        shuffle: bool = True,
    ):
        # 创建加权采样器
        weighted_sampler = CustomWeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=False)  # 不放回采样

        # 用分布式包装器包裹
        super(DistributedWeightedSampler, self).__init__(
            sampler=weighted_sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
