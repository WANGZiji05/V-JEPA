# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 默认 Collator —— 不做任何mask，直接返回原始数据
# ============================================================================
# 这个简单的collator用于不需要mask的场景（如纯分类任务的评估）。
# 它只是用PyTorch默认的collate函数将batch打包，返回空的mask列表。
#
# 与MultiBlock3D MaskCollator的区别：
# - DefaultCollator: 不做mask，返回 (data, None, None)
# - MaskCollator: 生成mask，返回 (data, masks_enc, masks_pred)

from logging import getLogger
import torch

_GLOBAL_SEED = 0
logger = getLogger()


class DefaultCollator(object):
    """
    默认的数据收集器 —— 不做任何遮罩处理

    直接使用PyTorch默认的collate函数打包batch，
    mask部分返回None（表示不遮罩，编码器可以看到全部内容）。

    使用场景：图像/视频分类评估时不需要mask，
    只需要将图像打包成batch即可。
    """
    def __call__(self, batch):
        # default_collate: 将列表中的样本堆叠成batch张量
        collated_batch = torch.utils.data.default_collate(batch)
        # 返回 (数据, None, None) —— 不需要mask
        return collated_batch, None, None
