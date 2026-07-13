# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 数据管理器 —— 统一的数据加载入口
# ============================================================================
# V-JEPA 支持多种数据集类型（视频数据集、图像数据集），
# 这个文件提供了一个统一的接口 init_data()，
# 根据配置自动选择正确的数据集类型并创建对应的DataLoader。
#
# 支持的数据类型：
# - VideoDataset: 视频数据集（预训练使用）
# - ImageNet/Places205/iNat21: 图像数据集（评估使用）

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,          # 数据增强transform
    shared_transform=None,   # 共享transform（视频评估时用）
    data='ImageNet',         # 数据集类型名称
    collator=None,           # 数据收集器（通常是mask生成器）
    pin_mem=True,            # 是否使用pinned memory
    num_workers=8,           # 数据加载worker数
    world_size=1,            # 分布式训练的world size
    rank=0,                  # 当前进程的rank
    root_path=None,          # 数据集根路径
    image_folder=None,       # 图像子文件夹
    training=True,           # 是否为训练模式
    copy_data=False,
    drop_last=True,
    tokenize_txt=True,
    subset_file=None,
    # 视频相关参数
    clip_len=8,              # 每个clip的帧数
    frame_sample_rate=2,     # 帧采样间隔
    duration=None,           # clip时长（秒）
    num_clips=1,             # 每个视频采样几个clip
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    decode_one_clip=True,
    datasets_weights=None,   # 各数据集采样权重
    persistent_workers=False,
    repeat_wds=False,
    ipe=300,
    log_dir=None,
):
    """
    统一的数据集初始化函数

    根据 data 参数自动选择正确的数据集类型：
    - 'imagenet' / 'inat21' / 'places205' → 图像数据集
    - 'videodataset' → 视频数据集（V-JEPA预训练使用）

    返回:
        (data_loader, dist_sampler): 数据加载器和分布式采样器
    """

    if (data.lower() == 'imagenet') \
            or (data.lower() == 'inat21') \
            or (data.lower() == 'places205'):
        # ---- 图像数据集（用于评估） ----
        from src.datasets.image_dataset import make_imagedataset
        dataset, data_loader, dist_sampler = make_imagedataset(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            copy_data=copy_data,
            drop_last=drop_last,
            subset_file=subset_file)

    elif data.lower() == 'videodataset':
        # ---- 视频数据集（用于预训练和视频评估） ----
        from src.datasets.video_dataset import make_videodataset
        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            frame_step=frame_sample_rate,
            duration=duration,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            log_dir=log_dir)

    # 只返回data_loader和dist_sampler（dataset不是必需的）
    return (data_loader, dist_sampler)
