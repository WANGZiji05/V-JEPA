# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# Mask 工具函数 —— 在 V-JEPA 中应用 mask 到 token 序列
# ============================================================================
# V-JEPA 的核心操作：给定一个完整的token序列，
# 根据mask索引只保留特定的token（丢弃被遮罩的token）。
#
# 这类似于 NLP 中只保留句子的某些单词，但在视觉领域：
# - encoder只看到context区域的token（未遮罩部分）
# - predictor需要context token + 目标区域的token（遮罩部分）
#
# 关键函数：torch.gather —— 根据索引从张量中提取元素

import torch


def apply_masks(x, masks, concat=True):
    """
    对token序列应用mask —— 只保留mask指定的token

    这是 V-JEPA 中非常关键的工具函数，被编码器和预测器频繁调用。

    参数:
        x: token张量，形状 [B, N, D]
           B = 批大小(batch size)
           N = 原始token总数（所有patch的数量）
           D = 特征维度
        masks: mask索引列表，每个mask形状 [B, K]
               K = 要保留的token数量（K << N，大部分token被丢弃）
               索引值在 [0, N) 范围内
        concat: 是否将多个mask的结果沿batch维度拼接

    返回:
        如果 concat=True: [len(masks)*B, K, D] 的张量（拼接所有mask的结果）
        如果 concat=False: 列表，每个元素形状 [B, K, D]

    工作原理：
    1. 对于每个mask，将其索引扩展为与x相同的特征维度
    2. 使用torch.gather根据索引从x中提取对应的token
    3. (可选)将所有mask的结果拼接

    torch.gather 的行为：
    - 对于dim=1（序列维度），根据index从x中选取元素
    - 结果形状: [B, K, D]（从每个样本中选出K个token）

    示例：
    x = [[a, b, c, d],   # 4个token，每个2维
         [e, f, g, h]]
    mask = [[0, 2],      # 保留第0和第2个token
            [1, 3]]      # 保留第1和第3个token
    结果 = [[a, c],
           [f, h]]
    """
    all_x = []
    for m in masks:
        # 将mask索引扩展为与x相同的形状，以便gather操作
        # m: [B, K] → unsqueeze(-1): [B, K, 1] → repeat: [B, K, D]
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        # torch.gather: 在dim=1上，根据mask_keep的索引选取token
        all_x += [torch.gather(x, dim=1, index=mask_keep)]

    if not concat:
        return all_x  # 返回列表，每个元素对应一种mask

    # 将所有mask的结果沿batch维度拼接
    # 例如: [B1,K,D] + [B2,K,D] → [(B1+B2), K, D]
    return torch.cat(all_x, dim=0)
