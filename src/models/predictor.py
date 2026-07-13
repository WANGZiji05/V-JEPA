# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 核心组件：预测器 (Predictor)
# ============================================================================
# 预测器是 V-JEPA 架构的关键创新之一。
#
# V-JEPA 的核心思想是 "Joint Embedding Predictive Architecture"（联合嵌入预测架构）：
# 1. 编码器(Encoder)看到视频的一部分（上下文 context）
# 2. 目标编码器(Target Encoder)看到视频的另一部分（目标 target，完整视频）
# 3. 预测器(Predictor)的任务：根据上下文特征，预测目标区域的特征
#
# 预测器的工作流程：
#   - 输入：上下文token（从编码器来的）+ 目标位置信息
#   - 内部：通过Transformer块处理，让上下文信息"传播"到目标位置
#   - 输出：预测的目标区域特征（与目标编码器输出的特征计算loss）
#
# V-JEPA的预测是在【特征空间/潜在空间】进行的，而不是像素空间！
# 这与MAE(Masked Autoencoder)等方法根本不同：
#   - MAE: 根据可见patch → 重建被遮罩patch的【像素值】
#   - V-JEPA: 根据可见patch → 预测被遮罩patch的【特征表示】
# 在特征空间预测更高效，因为不需要浪费算力去重建像素细节。

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import Block  # Transformer基础块
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks  # 对token应用mask


class VisionTransformerPredictor(nn.Module):
    """
    V-JEPA 预测器 —— 根据上下文特征预测目标区域的特征

    架构类似一个小型ViT，但有两个关键区别：
    1. 接收两种输入：上下文token + 目标token（或mask token）
    2. 输出只取目标token对应的部分，映射回编码器的特征维度

    预测器比编码器小得多：
    - 编码器通常是 ViT-L(1024维,24层) 或 ViT-H(1280维,32层)
    - 预测器通常只有 384维,6-12层
    这种不对称设计迫使预测器学习"理解"而非"复制"特征。

    参数说明：
        img_size: 输入空间尺寸
        patch_size: patch的空间尺寸
        num_frames: 视频帧数
        tubelet_size: 时间patch大小
        embed_dim: 编码器输出的特征维度（预测器输入维度）
        predictor_embed_dim: 预测器内部的特征维度（通常比embed_dim小）
        depth: 预测器Transformer的层数
        num_heads: 注意力头数
        use_mask_tokens: 是否使用可学习的mask token（而非目标编码器的输出）
        num_mask_tokens: mask token的种类数（对应不同mask策略）
        zero_init_mask_tokens: 是否将mask token初始化为零
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,             # 编码器输出维度（预测器的输入和输出维度）
        predictor_embed_dim=384,   # 预测器内部的工作维度（通常更小）
        depth=6,                   # 预测器的Transformer层数（通常比编码器浅）
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,     # 是否使用可学习的mask token
        num_mask_tokens=2,         # mask token的种类数
        zero_init_mask_tokens=True,  # mask token是否零初始化
        **kwargs
    ):
        super().__init__()
        # 【输入投影】将编码器特征映射到预测器的工作维度
        # 例如：1024维(编码器) → 384维(预测器)
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # 【Mask Token】（可选）
        # 当use_mask_tokens=True时，预测器不使用目标编码器的输出作为输入，
        # 而是使用可学习的mask token。这意味着预测器要"凭空"预测目标区域的特征。
        # num_mask_tokens对应不同的mask策略（因为一个样本可能被多种mask策略处理）
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for i in range(num_mask_tokens)
            ])

        # 保存几何参数
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        # 计算patch总数
        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.num_patches = num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )

        # 【位置编码】预测器有自己的位置编码
        # 因为上下文和目标token需要重新排列，位置信息不能从编码器继承
        self.uniform_power = uniform_power
        self.predictor_pos_embed = None
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False)

        # 【Transformer Blocks】预测器的核心处理层
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                attn_drop=attn_drop_rate,
                grid_size=grid_size,
                grid_depth=grid_depth,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 【输出投影】将预测器输出映射回编码器的特征维度
        # 例如：384维(预测器内部) → 1024维(与目标编码器输出对齐)
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # 权重初始化
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        """用sincos函数初始化预测器的位置编码"""
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
        """权重初始化（与编码器相同的策略）"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        """按层深度缩放残差连接（与编码器相同的策略）"""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):
        """
        【前向扩散过程】给目标特征添加噪声

        这是V-JEPA的一个重要设计：对目标特征进行扩散加噪。
        目的：防止预测器直接"复制"目标编码器的输出，
              迫使预测器学习更鲁棒的特征预测能力。

        扩散过程：
        1. 生成一个线性增加的噪声调度（beta从小到大）
        2. 随机采样一个时间步T
        3. 根据时间步T对应的alpha值，对特征进行加噪：
           z_noisy = √α * z + √(1-α) * ε    （ε是标准高斯噪声）

        参数:
            x: 目标编码器输出的特征
            noise_beta: 噪声强度的范围 (beta_start, beta_end)
            steps: 扩散步数
        """

        # 准备扩散噪声调度表
        b1, b2 = noise_beta
        beta_scheduler = (b1 + i*(b2-b1)/steps for i in range(steps))
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1.-_beta
            alpha_scheduler += [_alpha]

        # 为batch中的每个样本随机采样扩散时间步
        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)

        # 先归一化特征，再加噪
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha**0.5 * x + (1.-alpha)**0.5 * torch.randn(x.shape, device=x.device)
        return x

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=1):
        """
        预测器前向传播

        V-JEPA的核心操作图解：

                上下文区域                  目标区域
        ┌─────────────────┐        ┌─────────────────┐
        │ 已知的视频帧中的  │        │ 被遮罩的视频帧中的 │
        │ 可见区域         │        │ 不可见区域        │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 ▼                          ▼
           编码器输出                  目标编码器输出
          (context tokens)           (target tokens)
                 │                          │
                 │            ┌─────────────┘
                 │            │ (可选：添加扩散噪声)
                 │            ▼
                 │      噪声化的target tokens
                 │      或 可学习的mask token
                 │            │
                 ▼            ▼
            ┌─────────────────────┐
            │    预测器内部        │
            │  context tokens +   │
            │  target tokens      │
            │  通过Transformer    │
            │  让信息从context    │
            │  流向target位置     │
            └──────────┬──────────┘
                       │
                       ▼
                  预测的target区域特征
                  (与目标编码器输出计算L1/L2 Loss)

        参数:
            ctxt: 上下文token列表 [每个元素形状: B, N_ctxt, D]
                  从编码器输出，只包含未被遮罩区域的token
            tgt: 目标token列表 [每个元素形状: B, N_tgt, D]
                 从目标编码器输出
            masks_ctxt: 上下文token在原图中的位置索引
            masks_tgt: 目标token在原图中的位置索引（用于添加位置编码）
            mask_index: 使用哪个mask token（当use_mask_tokens=True时）
        """

        assert (masks_ctxt is not None) and (masks_tgt is not None), \
            'Cannot run predictor without mask indices'

        # 统一格式为列表
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        # 批大小（注意：一个样本可能被复制多次以支持多种mask策略）
        B = len(ctxt) // len(masks_ctxt)

        # ------------------------------------------------------------------- #
        # 步骤1：将上下文token映射到预测器的工作维度
        # 例如：1024维 → 384维
        # ------------------------------------------------------------------- #
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape

        # 添加上下文token的位置编码
        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(ctxt_pos_embed, masks_ctxt)

        # ------------------------------------------------------------------- #
        # 步骤2：准备目标区域的token
        # 有两种方式：
        #   A) 使用目标编码器输出 + 扩散噪声（让预测器学习去噪）
        #   B) 使用可学习的mask token（让预测器完全自主生成）
        # ------------------------------------------------------------------- #
        if self.mask_tokens is None:
            # 方式A：使用目标编码器输出，并添加扩散噪声
            pred_tokens = self.predictor_embed(tgt)  # 映射到预测器维度
            pred_tokens = self.diffusion(pred_tokens)  # 添加扩散噪声
        else:
            # 方式B：使用可学习的mask token
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]  # 选择对应mask策略的token
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)  # 扩展到所有patch
            pred_tokens = apply_masks(pred_tokens, masks_tgt)  # 只保留目标区域的token

        # 添加目标token的位置编码（告诉预测器这些token应该在什么位置）
        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_tgt)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs

        # ------------------------------------------------------------------- #
        # 步骤3：拼接上下文和目标token，一起通过Transformer处理
        # 上下文token在前，目标token在后
        # Transformer的自注意力会让信息从上下文区域流向目标区域
        # ------------------------------------------------------------------- #
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)  # [B, N_ctxt+N_tgt, D]

        # 合并mask矩阵（用于控制注意力计算）
        masks_ctxt = torch.cat(masks_ctxt, dim=0)
        masks_tgt = torch.cat(masks_tgt, dim=0)
        masks = torch.cat([masks_ctxt, masks_tgt], dim=1)

        # ------------------------------------------------------------------- #
        # 步骤4：通过预测器的Transformer块
        # ------------------------------------------------------------------- #
        for blk in self.predictor_blocks:
            x = blk(x, mask=masks)
        x = self.predictor_norm(x)

        # ------------------------------------------------------------------- #
        # 步骤5：只取目标token对应的输出，映射回编码器维度
        # 前面的N_ctxt个token是上下文，我们只关心预测的目标区域
        # ------------------------------------------------------------------- #
        x = x[:, N_ctxt:]  # 切掉上下文token部分，只保留目标区域预测
        x = self.predictor_proj(x)  # 映射回编码器维度（例如：384维 → 1024维）

        return x


def vit_predictor(**kwargs):
    """创建V-JEPA预测器的工厂函数"""
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
