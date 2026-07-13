# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 图像数据集加载器 —— 用于 V-JEPA 的图像分类评估
# ============================================================================
# V-JEPA 虽然是在视频上预训练的，但评估也包含图像分类任务
# （如 ImageNet、Places205、iNaturalist）。
#
# 本文件基于 PyTorch 的 ImageFolder 类，实现了简单的图像数据集加载。
# 图像目录结构：
#   root_path/image_folder/train/class_1/xxx.png
#   root_path/image_folder/train/class_2/yyy.png
#   ...
#   root_path/image_folder/val/class_1/zzz.png

import os
from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


class ImageFolder(torchvision.datasets.ImageFolder):
    """
    扩展 PyTorch 的 ImageFolder

    添加了对自定义数据路径的支持：
    自动在 root_path 下拼接 image_folder 和 train/val 子目录。

    目录结构示例：
    root_path = "/data/"
    image_folder = "imagenet_full_size/061417/"
    → 训练数据: /data/imagenet_full_size/061417/train/
    → 验证数据: /data/imagenet_full_size/061417/val/
    """
    def __init__(
        self,
        root,                    # 数据集根路径
        image_folder='imagenet_full_size/061417/',  # 图片子目录
        transform=None,          # 数据增强
        train=True,              # True=训练集, False=验证集
    ):
        # 拼接完整的图片路径
        suffix = 'train/' if train else 'val/'
        data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')
        # 调用父类初始化
        super(ImageFolder, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageFolder')


def make_imagedataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    persistent_workers=False,
    subset_file=None
):
    """
    创建图像数据集和DataLoader

    与make_videodataset类似，但用于图像数据。
    主要用于V-JEPA的冻结评估（frozen evaluation）。

    流程：
    1. 创建ImageFolder数据集
    2. 创建分布式采样器
    3. 创建DataLoader
    """
    # 创建数据集
    dataset = ImageFolder(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training)
    logger.info('ImageFolder dataset created')

    # 分布式采样器
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    # DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers)
    logger.info('ImageFolder unsupervised data loader created')

    return dataset, data_loader, dist_sampler
