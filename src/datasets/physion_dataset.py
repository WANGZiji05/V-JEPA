# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# Physion++ 数据集加载器
# ============================================================================
#
# Physion++ 是一个用于评估物理场景理解能力的数据集。
# 任务 (OCP - Object Contact Prediction): 预测两个物体是否会接触。
#
# 【数据集包含 4 种物理属性】
#   - mass           (质量)
#   - friction       (摩擦力)
#   - elasticity     (弹性，对应 bouncy_* 场景)
#   - deformability  (可变形性)
#
# 【数据划分为三个子集】
#   - data_v1:          训练集 (用于自监督预训练或探针训练)
#   - readout_data_v1:  读出色合 (用于训练探针)
#   - testdata_v1:      测试集 (配对试验，用于最终评估)
#
# 【实际目录结构】
#   data_v1/
#   ├── bouncy_wall_pp/              # scenario 文件夹
#   │   └── bouncy_wall-zld=0-.../   # config 子文件夹
#   │       ├── 0000_img.mp4          # RGB 视频 (V-JEPA 使用这个)
#   │       ├── 0000_id.mp4           # 分割掩码视频 (忽略)
#   │       └── 0000.pkl              # 元数据 (含 OCP 标签)
#   ├── friction_collision_pp/
#   ├── mass_dominoes_pp/
#   └── ...
#
#   testdata_v1/ 额外包含 merge CSV 文件存放标签：
#     physionpp-bouncy_merge_*.csv   → elasticity 标签
#     physionpp-deform_merge_*.csv   → deformability 标签
#     physionpp-friction_merge_*.csv → friction 标签
#     physionpp-mass_merge_*.csv     → mass 标签
#
# 【配对试验 (Paired Trials) 设计】
#   测试集中每个 scenario 都有 -copy0 和 -copy1 两个版本。
#   两个 trial 在预测阶段的初始帧看起来完全一样，
#   但因为底层物理属性值不同，最终结果（接触/不接触）不同。
#   这迫使模型学习真正的物理理解，而非依赖表面的视觉线索。
#
# 【CSV 索引格式】
#   由 scripts/prepare_physion_data.py 生成。每行格式:
#     /absolute/path/to/0000_img.mp4,property_name,label
#     - property_name: 'mass' | 'friction' | 'elasticity' | 'deformability'
#     - label: 0 (不会接触, NO) | 1 (会接触, YES) | -1 (无标签)
# ============================================================================

import os
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch

from decord import VideoReader, cpu

_GLOBAL_SEED = 0
logger = getLogger()

# Physion++ 的 4 种物理属性
PHYSION_PROPERTIES = ['mass', 'friction', 'elasticity', 'deformability']

# 属性名称到索引的映射
PROPERTY_TO_INDEX = {prop: i for i, prop in enumerate(PHYSION_PROPERTIES)}


def make_physion_dataset(
    data_paths,
    batch_size,
    frames_per_clip=16,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
    properties=None,
    return_property_label=True,
):
    """
    创建 Physion++ 数据集和数据加载器的工厂函数。

    与 make_videodataset() 保持一致的设计模式，
    但额外支持物理属性标签和多属性训练。

    参数:
        data_paths (list[str]): CSV 索引文件路径列表
        batch_size (int): 每个 GPU 的批次大小
        frames_per_clip (int): 每个 clip 的帧数
        frame_step (int): 帧采样间隔
        num_clips (int): 每个视频采样的 clip 数
        random_clip_sampling (bool): 是否随机采样 clip 位置
        allow_clip_overlap (bool): 是否允许 clip 重叠
        filter_short_videos (bool): 过滤过短的视频
        filter_long_videos (int): 过滤过长的视频（字节数上限）
        transform: 每个 clip 独立的数据增强
        shared_transform: 所有 clip 共享的数据增强
        rank (int): 当前进程的 rank
        world_size (int): 分布式训练的 world size
        collator: 数据收集器
        drop_last (bool): 是否丢弃最后不完整的 batch
        num_workers (int): 数据加载的 worker 数
        pin_mem (bool): 是否使用 pinned memory
        duration (float): clip 时长（秒），None 则用 frames_per_clip
        log_dir (str): 日志输出路径
        properties (list[str]): 要加载的物理属性子集
            None 表示加载所有 4 种属性
            例如 ['mass', 'friction'] 只加载质量和摩擦力数据
        return_property_label (bool): 是否返回物理属性标签
            True: 返回 (frames, ocp_label, property_label)
            False: 返回 (frames, ocp_label)

    返回:
        tuple: (data_loader, dist_sampler)
    """

    dataset = PhysionDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform,
        properties=properties,
        return_property_label=return_property_label,
    )

    logger.info(f'PhysionDataset created with {len(dataset)} samples')

    # 分布式采样器
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    # DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    logger.info('PhysionDataset data loader created')

    return data_loader, dist_sampler


class PhysionDataset(torch.utils.data.Dataset):
    """
    Physion++ 视频数据集类。

    负责:
    1. 从 CSV 文件读取视频路径和标签（OCP 标签 + 物理属性标签）
    2. 使用 decord 解码视频文件
    3. 从视频中采样 clip（连续的帧序列）
    4. 应用数据增强

    【CSV 格式】
    每行: /path/to/video.mp4,property_name,binary_label
    示例: /data/physion/train/mass/trial_0000.mp4,mass,1

    【返回格式】
    return_property_label=True:
        (frames, ocp_label, property_label)
    return_property_label=False:
        (frames, ocp_label)

    【多属性训练】
    当 properties=None 时，加载所有 4 种物理属性的数据。
    可以指定 properties=['mass', 'friction'] 只训练特定属性。
    """

    def __init__(
        self,
        data_paths,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,
        properties=None,
        return_property_label=True,
    ):
        self.data_paths = data_paths
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
        self.return_property_label = return_property_label

        if VideoReader is None:
            raise ImportError(
                'Unable to import "decord" which is required to read videos.'
            )

        # ---- 从 CSV 文件中加载视频路径和标签 ----
        samples = []
        ocp_labels = []
        property_labels = []
        self.num_samples_per_dataset = []

        for data_path in self.data_paths:
            if data_path.endswith('.csv'):
                # CSV 格式: video_path,property_name,ocp_label
                data = pd.read_csv(data_path, header=None)
                # 检查 CSV 列数：支持 3 列标准格式
                if data.shape[1] >= 3:
                    paths = list(data.values[:, 0])
                    props = list(data.values[:, 1])
                    labels = list(data.values[:, 2])
                elif data.shape[1] == 2:
                    # 兼容旧格式: video_path,ocp_label (不区分属性)
                    paths = list(data.values[:, 0])
                    props = ['unknown'] * len(paths)
                    labels = list(data.values[:, 1])
                else:
                    raise ValueError(
                        f'CSV must have at least 2 columns, got {data.shape[1]}'
                    )

                # 按属性过滤
                for p, prop, lbl in zip(paths, props, labels):
                    prop_str = str(prop).strip().lower()
                    if properties is not None and prop_str not in properties:
                        continue
                    samples.append(str(p).strip())
                    ocp_labels.append(int(lbl))
                    property_labels.append(PROPERTY_TO_INDEX.get(
                        prop_str, -1
                    ))

                self.num_samples_per_dataset.append(len(samples))
            else:
                logger.warning(f'Unsupported file format: {data_path}')

        self.samples = samples
        self.ocp_labels = ocp_labels
        self.property_labels = property_labels

        # 统计各属性样本数
        if len(property_labels) > 0:
            unique, counts = np.unique(property_labels, return_counts=True)
            prop_stats = ', '.join(
                f'{PHYSION_PROPERTIES[u]}: {c}'
                for u, c in zip(unique, counts)
                if u >= 0
            )
            logger.info(f'Physion++ dataset composition: {prop_stats}')

    def __getitem__(self, index):
        """
        获取一个训练/评估样本。

        返回:
            buffer: 视频帧 tensor [num_clips, C, frames_per_clip, H, W]
            ocp_label: OCP 标签 (int, 0=NO, 1=YES, -1=unknown)
            (可选) property_label: 物理属性标签 (int, 0-3)
        """
        sample = self.samples[index]
        ocp_label = self.ocp_labels[index]
        property_label = self.property_labels[index]

        # 尝试加载视频（视频损坏则随机换一个）
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices = self.loadvideo_decord(sample)
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]
                ocp_label = self.ocp_labels[index]
                property_label = self.property_labels[index]

        def split_into_clips(video):
            """将视频帧序列切分成多个 clip"""
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i * fpc:(i + 1) * fpc] for i in range(nc)]

        # 应用 transform 流水线
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        if self.return_property_label:
            return buffer, ocp_label, property_label
        else:
            return buffer, ocp_label

    def loadvideo_decord(self, sample):
        """
        使用 decord 加载视频并采样帧。

        采样逻辑与 VideoDataset.loadvideo_decord() 保持一致。
        详见 src/datasets/video_dataset.py 的注释。
        """
        fname = sample

        # 检查文件是否存在
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        # 检查文件大小
        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        # 打开视频
        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)

        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((
                        indices,
                        np.ones(fpc - partition_len // fstp) * partition_len,
                    ))
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((
                        indices,
                        np.ones(fpc - sample_len // fstp) * sample_len,
                    ))
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
