# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 正弦-余弦位置编码 (Sinusoidal Position Embedding)
# ============================================================================
# Transformer本身对token的处理是"位置无关"的——无论token在什么位置，
# 自注意力的计算方式都一样。为了让模型知道每个token在图像/视频中的位置，
# 我们需要为每个token添加位置编码。
#
# 正弦-余弦位置编码的公式（原始Transformer论文提出）：
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#   其中 pos 是位置索引，i 是特征维度的索引，d 是总维度
#
# 为什么用sin/cos？
# 1. 相邻位置有相似的编码（连续性）
# 2. 不同维度使用不同频率（从高频到低频），能捕捉不同尺度的位置关系
# 3. 可以外推到训练时未见过的位置（因为sin/cos是连续函数）
# 4. 不需要训练参数（节省显存，加速训练）
#
# 2D位置编码 = 高度编码(前D/2维) + 宽度编码(后D/2维)
# 3D位置编码 = 时间深度编码(前D/2维) + 高度编码(D/4维) + 宽度编码(D/4维)
#              时间维度占用更多维度，因为时序信息对视频理解更重要

import numpy as np


def get_3d_sincos_pos_embed(
    embed_dim,        # 总编码维度
    grid_size,        # 空间网格大小（高度和宽度方向的patch数）
    grid_depth,       # 时间深度（时间方向的tubelet数）
    cls_token=False,  # 是否为CLS token预留位置
    uniform_power=False  # 是否均匀分配各维度的频率
):
    """
    生成3D正弦-余弦位置编码（用于视频）

    输出形状: [grid_depth * grid_size * grid_size, embed_dim]

    各维度分配（non-uniform，默认）：
    - 时间深度维: embed_dim / 2 个维度
    - 高度维:     embed_dim / 4 个维度
    - 宽度维:     embed_dim / 4 个维度
    （时间维度占一半，因为视频中时序信息更关键）

    各维度分配（uniform）：
    - 三维各占 embed_dim / 3（大约）
    """
    # 生成三个维度的坐标网格
    grid_d = np.arange(grid_depth, dtype=float)  # 时间坐标: [0, 1, ..., grid_depth-1]
    grid_h = np.arange(grid_size, dtype=float)   # 高度坐标: [0, 1, ..., grid_size-1]
    grid_w = np.arange(grid_size, dtype=float)   # 宽度坐标: [0, 1, ..., grid_size-1]

    # np.meshgrid生成3D网格坐标
    # 注意：meshgrid的顺序很重要！这里用[d, h, w]的顺序
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    # 分配各维度的编码维度
    if not uniform_power:
        # 默认：时间占1/2，空间各占1/4
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        # 均匀分配：三维各占1/3
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim/6)*2)

    # 为每个维度分别生成1D位置编码，然后拼接
    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # 高度编码
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # 宽度编码
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # 时间深度编码
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # 拼接三个维度的编码
    pos_embed = pos_embed[:, :embed_dim]  # 确保总维度正确

    if cls_token:
        # 如果使用CLS token，在最前面加一个全零的位置编码
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    生成2D正弦-余弦位置编码（用于图像）

    输出形状: [grid_size * grid_size, embed_dim]
    编码维度：前D/2为高度编码，后D/2为宽度编码
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    # np.meshgrid生成2D网格
    # 注意顺序：[w, h] → h在第二维，w在第一维
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)

    # 高度和宽度各占一半维度
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # 拼接

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    生成1D正弦-余弦位置编码（用于一维序列）

    输出形状: [grid_size, embed_dim]
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从位置数组生成1D正弦-余弦位置编码

    这是位置编码的核心实现：
    对于每个位置pos：
      PE(pos, 2i)   = sin(pos / 10000^(2i/d))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    参数:
        embed_dim: 输出编码的维度（必须是偶数）
        pos: 位置数组，形状 [M]

    返回:
        编码矩阵，形状 [M, embed_dim]
        其中偶数列(0,2,4,...)是sin值，奇数列(1,3,5,...)是cos值

    关键理解：
    - omega 是从低频到高频的频率序列: [1/1, 1/2.15, 1/4.64, ..., 1/10000]
    - pos * omega 是外积，为每个位置×每个频率计算一个值
    - 最后对一半维度取sin，一半取cos（产生相位偏移）
    """
    assert embed_dim % 2 == 0  # 维度必须是偶数

    # 计算频率序列 omega
    # omega = 1 / (10000^(2i/d))  其中 i = 0, 1, ..., d/2-1
    omega = np.arange(embed_dim // 2, dtype=float)  # [0, 1, 2, ..., d/2-1]
    omega /= embed_dim / 2.       # [0, 2/d, 4/d, ..., (d-2)/d]
    omega = 1. / 10000**omega     # [1/1, 1/2.15, 1/4.64, ..., 1/10000]

    # 外积：每个位置 × 每个频率
    # pos: [M] → 维度扩展后 → [M, 1]
    # omega: [D/2] → 维度扩展后 → [1, D/2]
    # 外积结果: [M, D/2]
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)  # 比循环快得多的外积运算

    # 一半维度用sin，一半维度用cos
    emb_sin = np.sin(out)  # 偶数列
    emb_cos = np.cos(out)  # 奇数列

    # 拼接：[M, D/2] + [M, D/2] = [M, D]
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
