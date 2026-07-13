# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 视频数据集加载器 —— V-JEPA 的数据来源
# ============================================================================
# V-JEPA 使用大规模视频数据集进行预训练（如 VideoMix2M）。
# 本文件实现了视频数据的读取、采样、切分和多clip采样。
#
# 数据格式要求：
# CSV文件，每行格式: `/path/to/video.mp4 $integer_class_label`
# 预训练时类别标签被忽略（自监督学习不需要标签），
# 评估时需要真实的类别标签。
#
# 视频读取使用 decord 库（高效的视频解码器）。

import os
import pathlib
import warnings
from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu  # decord: 高效的视频解码库

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,              # 数据集CSV文件路径列表
    batch_size,
    frames_per_clip=8,       # 每个clip的帧数
    frame_step=4,            # 帧采样间隔（每隔多少帧取一帧）
    num_clips=1,             # 每个视频采样几个clip
    random_clip_sampling=True,  # 是否随机采样clip位置
    allow_clip_overlap=False,   # 是否允许clip之间重叠
    filter_short_videos=False,  # 是否过滤太短的视频
    filter_long_videos=int(10**9),  # 过滤太长的视频（字节数上限）
    transform=None,            # 数据增强transform
    shared_transform=None,     # 所有clip共享的transform
    rank=0,
    world_size=1,
    datasets_weights=None,     # 各数据集的采样权重（用于不平衡数据集）
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,             # clip的时长（秒）
    log_dir=None,
):
    """创建视频数据集和数据加载器"""

    # 创建数据集对象
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('VideoDataset dataset created')

    # 创建分布式采样器（确保每个GPU处理不同的数据）
    if datasets_weights is not None:
        # 有数据集权重时使用加权采样器
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,  # 总GPU数
            rank=rank,               # 当前GPU排名
            shuffle=True)
    else:
        # 标准的分布式采样器
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    # 创建DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,  # mask生成器在这里被调用
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,   # pinned memory加速CPU→GPU数据传输
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """
    视频数据集类

    负责：
    1. 从CSV文件读取视频路径列表
    2. 解码视频文件
    3. 从视频中采样clip（连续的帧序列）
    4. 将clip切分成多个子clip（支持多clip采样）
    5. 应用数据增强
    """

    def __init__(
        self,
        data_paths,              # 数据集CSV文件路径列表
        datasets_weights=None,   # 各数据集权重
        frames_per_clip=16,      # 每个clip包含多少帧
        frame_step=4,            # 帧采样间隔
        num_clips=1,             # 每个视频采样几个clip
        transform=None,          # 每个clip独立的transform
        shared_transform=None,   # 所有clip共享的transform
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,           # clip时长（秒），用于动态计算采样率
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # 从所有CSV文件中读取视频路径和标签
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path[-4:] == '.csv':
                # CSV格式: 每行 "视频路径 类别标签"
                data = pd.read_csv(data_path, header=None, delimiter=" ")
                samples += list(data.values[:, 0])   # 视频路径
                labels += list(data.values[:, 1])    # 类别标签
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)
            elif data_path[-4:] == '.npy':
                # NPY格式: 直接是视频路径数组
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)  # 预训练时不关心标签
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

        # 计算每个样本的采样权重（用于不平衡数据集）
        # 权重 = 数据集权重 / 该数据集的样本数
        # 这样每个数据集的样本被采样到的总概率与数据集权重成正比
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        """
        获取一个训练样本

        返回:
            buffer: 视频帧列表 [num_clips × frames_per_clip]
            label: 类别标签（预训练时不用）
            clip_indices: 每个clip的帧索引
        """
        sample = self.samples[index]

        # 尝试加载视频（如果视频损坏则随机换一个）
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices = self.loadvideo_decord(sample)  # [T, H, W, 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        label = self.labels[index]

        def split_into_clips(video):
            """将视频帧序列切分成多个clip"""
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # 应用transform
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)    # 共享transform（先处理完整帧序列）
        buffer = split_into_clips(buffer)              # 切分成clip
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]  # 每个clip独立的transform

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample):
        """
        使用 decord 库加载视频

        采样策略：
        1. 将视频分成 num_clips 等份
        2. 从每份中随机采样一个clip（或固定位置采样）
        3. 每个clip包含 frames_per_clip 帧，帧之间间隔 frame_step

        例如：fpc=16, fstp=4, num_clips=2
          需要16帧 × 4间隔 = 64帧的窗口
          视频被分成2段，每段内随机选一个64帧的窗口
        """
        fname = sample

        # 检查文件是否存在
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        # 检查文件大小
        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # 太小的文件跳过
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:  # 太大的文件跳过
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        # 用decord打开视频
        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        # 计算实际需要的帧参数
        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            # 如果指定了clip时长，根据视频帧率计算采样间隔
            try:
                fps = vr.get_avg_fps()  # 获取视频帧率
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)  # 一个clip需要的总帧跨度

        # 过滤太短的视频
        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)  # 回到视频开头

        # 将视频等分成 num_clips 段
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                # 情况1：分段长度 > 所需窗口长度
                # 在分段内随机采样clip的起始位置
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                indices = indices + i * partition_len  # 偏移到对应分段
            else:
                # 情况2：分段长度 < 所需窗口长度
                if not self.allow_clip_overlap:
                    # 不允许重叠：重复最后一帧来凑够帧数
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    # 允许重叠：clip之间可以重叠
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        # 一次性读取所有需要的帧（比逐帧读取快得多）
        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
