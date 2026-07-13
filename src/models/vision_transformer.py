# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 核心组件：Vision Transformer (ViT) 编码器
# ============================================================================
# ViT 是 V-JEPA 的基础骨干网络，用于将视频/图像转换为特征表示（embedding）。
# 核心思想：将图像/视频切分成小块(patch)，然后把每个patch当作NLP中的"单词"，
# 用 Transformer 的自注意力机制来学习这些patch之间的关系。
#
# 对于视频输入：使用 3D patch（管状 tubelet），同时在时间和空间维度上切分。
# 对于图像输入：使用 2D patch，只在空间维度上切分。

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D  # 将原始像素转为patch embedding
from src.models.utils.modules import Block  # Transformer的一个基础块（注意力+MLP）
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed  # 生成位置编码
from src.utils.tensors import trunc_normal_  # 截断正态分布初始化
from src.masks.utils import apply_masks  # 对token应用mask（保留/移除某些patch）


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) —— V-JEPA 的编码器骨干网络

    ViT 的工作流程：
    1. 【Patch Embedding】将输入图像/视频切分成固定大小的小块，每个小块通过卷积映射为一个向量(token)
    2. 【位置编码】为每个patch添加位置信息（因为Transformer本身不感知位置）
    3. 【Transformer Blocks】多个Transformer块串联，通过自注意力机制处理所有token
    4. 【输出】得到每个patch对应的特征向量

    参数说明：
        img_size: 输入图像/视频的空间尺寸（默认224x224像素）
        patch_size: 每个patch的空间尺寸（默认16x16像素）
        num_frames: 视频帧数（>1表示视频，=1表示图像）
        tubelet_size: 时间维度上的patch大小（视频中连续tubelet_size帧组成一个tubelet）
        in_chans: 输入通道数（RGB彩色图像=3）
        embed_dim: 特征向量维度（token的维度，越大模型越强但越慢）
        depth: Transformer块的层数（越深模型越强但越慢）
        num_heads: 多头注意力的头数
        mlp_ratio: MLP隐藏层维度相对于embed_dim的倍数
        out_layers: 指定哪些层的输出需要返回（用于中间层特征提取）
        uniform_power: 是否在3D位置编码中均匀分配各维度的频率
    """
    def __init__(
        self,
        img_size=224,           # 输入空间分辨率
        patch_size=16,          # patch的空间尺寸
        num_frames=1,           # 输入帧数（1=图像模式，>1=视频模式）
        tubelet_size=2,         # 时间维度的patch大小（用于3D patch）
        in_chans=3,             # 输入通道数（RGB=3）
        embed_dim=768,          # token的特征向量维度
        depth=12,               # Transformer层数
        num_heads=12,           # 多头注意力头数
        mlp_ratio=4.0,          # MLP隐藏层倍数（hidden_dim = embed_dim * mlp_ratio）
        qkv_bias=True,          # QKV投影是否使用偏置
        qk_scale=None,          # QK点积的缩放因子（None则自动为 1/√head_dim）
        drop_rate=0.0,          # Dropout比率
        attn_drop_rate=0.0,     # 注意力权重的Dropout比率
        norm_layer=nn.LayerNorm,  # 归一化层类型
        init_std=0.02,          # 权重初始化的标准差
        out_layers=None,        # 需要额外返回中间层输出的层索引列表
        uniform_power=False,    # 是否在3D位置编码中均匀分配频率
        **kwargs
    ):
        super().__init__()
        # 保存关键参数
        self.num_features = self.embed_dim = embed_dim  # 输出特征的维度
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1  # 判断是视频还是图像模式

        # 计算patch网格的尺寸
        # 例如：224x224图像，patch_size=16 → grid_size=14（每行/列14个patch）
        grid_size = self.input_size // self.patch_size
        # 例如：16帧视频，tubelet_size=2 → grid_depth=8（时间维度上8个tubelet）
        grid_depth = self.num_frames // self.tubelet_size

        # ------------------------------------------------------------------- #
        # 第1步：【Patch Embedding】用卷积将原始像素转换为token序列
        # 视频：使用3D卷积 (tubelet_size, patch_size, patch_size) 的核
        #       一次处理 tubelet_size 帧 × patch_size 高 × patch_size 宽 的时空块
        # 图像：使用2D卷积 (patch_size, patch_size) 的核
        # ------------------------------------------------------------------- #
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            # 视频patch总数 = 时间深度 × 高度patch数 × 宽度patch数
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            # 图像patch总数 = 高度patch数 × 宽度patch数
            self.num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )

        # ------------------------------------------------------------------- #
        # 第2步：【位置编码】为每个patch添加位置信息
        # 使用正弦-余弦位置编码（不需要学习，从三角函数公式计算）
        # requires_grad=False 表明位置编码不会被训练更新
        # ------------------------------------------------------------------- #
        self.uniform_power = uniform_power
        self.pos_embed = None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False)  # 位置编码是固定的，不参与梯度更新

        # ------------------------------------------------------------------- #
        # 第3步：【Transformer Blocks】多个Transformer层串联
        # 每一层包含：多头自注意力(MSA) + 多层感知机(MLP)
        # 使用残差连接和LayerNorm归一化
        # grid_size/grid_depth 用于某些注意力变体（非标准ViT可忽略）
        # ------------------------------------------------------------------- #
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,     # GELU激活函数（比ReLU更平滑）
                grid_size=grid_size,
                grid_depth=grid_depth,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])  # 创建depth个完全相同的Transformer块
        self.norm = norm_layer(embed_dim)  # 最后的LayerNorm层

        # ------------------------------------------------------------------- #
        # 第4步：【权重初始化】
        # - 位置编码用sincos公式初始化
        # - 线性层用截断正态分布初始化
        # - 对深层网络的残差路径进行缩放，防止训练不稳定
        # ------------------------------------------------------------------- #
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # 用sincos初始化位置编码
        self.init_std = init_std
        self.apply(self._init_weights)  # 对所有子模块应用权重初始化
        self._rescale_blocks()  # 按层深度缩放残差路径

    def _init_pos_embed(self, pos_embed):
        """
        用正弦-余弦函数初始化位置编码
        2D位置编码 = 高度方向sincos + 宽度方向sincos（各占一半维度）
        3D位置编码 = 深度(时间)方向sincos + 高度方向sincos + 宽度方向sincos
        时间维度占一半特征维度，空间两维各占四分之一
        """
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        """
        权重初始化策略：
        - nn.Linear: 截断正态分布 N(0, init_std)，偏置初始化为0
        - nn.LayerNorm: 偏置=0，权重=1
        - nn.Conv2d/Conv3d: 截断正态分布 N(0, init_std)
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """
        对深层Transformer的残差连接进行缩放
        将注意力投影层和MLP输出层的权重除以 √(2*layer_id)
        这样做的作用：防止深层网络的输出方差过大，有助于训练稳定性
        （类似于GPT-2中的残差缩放技巧，但这里是按层深度缩放）
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        """返回Transformer的层数"""
        return len(self.blocks)

    def no_weight_decay(self):
        """返回不需要权重衰减的参数（ViT编码器没有这样的参数）"""
        return {}

    def forward(self, x, masks=None):
        """
        前向传播（V-JEPA中的核心操作）

        参数:
            x: 输入张量
               - 图像: [B, C, H, W]  批大小, 通道数, 高度, 宽度
               - 视频: [B, C, T, H, W] 批大小, 通道数, 帧数, 高度, 宽度
            masks: 可选，要保留的patch索引列表
                   每个mask的形状为 [B, K]，其中K是要保留的patch数量
                   V-JEPA中用于遮罩掉某些patch，只让编码器看到上下文(context)部分

        返回:
            经过Transformer处理后的特征张量 [B, N, D]
            如果指定了out_layers，返回中间层的输出列表
        """

        # 统一将mask转换为列表格式（支持单个mask或多个mask并行处理）
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # ------------------------------------------------------------------- #
        # 第1步：将输入转为patch token序列
        # 图像: [B, 3, 224, 224] → [B, 196, 768]  (196 = 14×14个patch)
        # 视频: [B, 3, 16, 224, 224] → [B, 1568, 1280] (1568 = 8×14×14)
        # ------------------------------------------------------------------- #
        pos_embed = self.pos_embed
        if pos_embed is not None:
            # 如果输入分辨率与训练时不同，对位置编码进行插值
            pos_embed = self.interpolate_pos_encoding(x, pos_embed)
        x = self.patch_embed(x)  # 卷积投影：像素空间 → 特征空间
        if pos_embed is not None:
            x += pos_embed  # 加上位置编码（广播加法：[1, N, D] + [B, N, D]）
        B, N, D = x.shape  # B=批大小, N=patch数, D=特征维度

        # ------------------------------------------------------------------- #
        # 第2步（V-JEPA特有）：应用mask，只保留上下文(context)区域的patch
        # apply_masks使用torch.gather操作，根据索引提取需要的patch
        # ------------------------------------------------------------------- #
        if masks is not None:
            x = apply_masks(x, masks)  # 只保留mask指定的patch
            masks = torch.cat(masks, dim=0)  # 合并多个mask（用于注意力计算中的mask）

        # ------------------------------------------------------------------- #
        # 第3步：通过所有Transformer块进行前向传播
        # 每个Block内部：x = x + Attention(Norm(x)) → x = x + MLP(Norm(x))
        # 如果指定了out_layers，保存这些中间层的输出
        # ------------------------------------------------------------------- #
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks)  # mask可用于控制哪些token之间可以互相注意
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))  # 保存中间层归一化后的输出

        if self.out_layers is not None:
            return outs  # 返回多个中间层的输出

        if self.norm is not None:
            x = self.norm(x)  # 最后的LayerNorm归一化

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        """
        当输入尺寸与预训练时不同时，对位置编码进行插值

        这是ViT的一个重要技巧：预训练模型可能在224x224上训练，
        但在测试时可以使用不同分辨率（如384x384），此时需要插值位置编码。

        视频模式：使用三线性插值（时间+高度+宽度）
        图像模式：使用双三次插值（高度+宽度）
        """

        _, N, dim = pos_embed.shape

        if self.is_video:
            # ---- 视频模式 ----
            _, _, T, H, W = x.shape
            # 如果尺寸匹配，直接返回
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed

            # 将像素/帧单位转换为patch/tubelet单位
            T = T // self.tubelet_size  # 时间维度的patch数
            H = H // self.patch_size    # 高度维度的patch数
            W = W // self.patch_size    # 宽度维度的patch数

            # 计算预训练时的patch网格尺寸
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

            # 计算插值比例
            scale_factor = (T/N_t, H/N_h, W/N_w)

            # 使用三线性插值（trilinear）调整位置编码的尺寸
            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:
            # ---- 图像模式 ----
            _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size:
                return pos_embed

            # 计算patch数量和插值因子
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            # 使用双三次插值（bicubic）调整位置编码的尺寸
            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


# ============================================================================
# 以下是一些预定义的ViT模型配置（从小到大的不同规模）
# 命名规则: vit_<size>  eg. vit_tiny, vit_small, vit_base, vit_large, vit_huge
#
# 模型规模对比:
#   tiny:    embed_dim=192,  depth=12,  heads=3   （最小，适合快速实验）
#   small:   embed_dim=384,  depth=12,  heads=6
#   base:    embed_dim=768,  depth=12,  heads=12  （类似ViT-B/16）
#   large:   embed_dim=1024, depth=24,  heads=16  （V-JEPA论文中用到的L模型）
#   huge:    embed_dim=1280, depth=32,  heads=16  （V-JEPA论文中用到的H模型）
#   giant:   embed_dim=1408, depth=40,  heads=16
#   gigantic: embed_dim=1664, depth=48, heads=16  （最大，需要很多GPU）
# ============================================================================

def vit_tiny(patch_size=16, **kwargs):
    """ViT-Tiny: 最小规模的ViT，适合快速实验和调试"""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    """ViT-Small: 小型ViT"""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    """ViT-Base: 标准规模的ViT，类似BERT-base"""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    """ViT-Large: 大规模ViT（V-JEPA论文中使用的L配置）"""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    """ViT-Huge: 超大规模ViT（V-JEPA论文中使用的H配置，性能最强）"""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    """ViT-Giant: 巨型ViT"""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_gigantic(patch_size=14, **kwargs):
    """ViT-Gigantic: 超巨型ViT，使用更小的patch_size(14)"""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1664, depth=48, num_heads=16, mpl_ratio=64/13,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


# 各模型规模对应的特征维度字典
VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
    'vit_gigantic': 1664,
}
