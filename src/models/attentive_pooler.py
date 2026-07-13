# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
================================================================================
V-JEPA 注意力汇聚器与分类器 (Attentive Pooler & Classifier)
================================================================================

本模块实现了 V-JEPA 中用于"从视频特征序列中提取单一全局表示"的核心组件。

背景与动机：
    V-JEPA 的视频编码器（如 Vision Transformer）输出的不是一个向量，
    而是一系列 patch 级别的特征向量（例如输入视频被切成了 N 个小块，
    则输出是 N 个特征向量）。但在下游评估任务（如视频分类）中，
    我们通常需要**一个固定大小的向量**来表示整个视频。

    传统的做法是对所有 patch 特征做简单的平均池化（Mean Pooling），
    但这样做忽略了不同 patch 对任务的贡献不同 ——

    举个例子：
        要判断视频是"打篮球"还是"踢足球"，画面中有球的那几个 patch
        应该比背景天空的 patch 更受关注。

    注意力汇聚器 (Attentive Pooler) 通过**可学习的查询向量 (Query Tokens)**
    和**交叉注意力 (Cross-Attention)** 机制，自动学习"关注哪些 patch"。

模块包含两个类：
  1. AttentivePooler      — 注意力汇聚器：将 N 个 patch 特征聚合成 1 个全局特征
  2. AttentiveClassifier  — 注意力分类器：在汇聚器之上加一个线性分类头
================================================================================
"""

import math

import torch
import torch.nn as nn

from src.models.utils.modules import (
    Block,
    CrossAttention,
    CrossAttentionBlock
)
from src.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """
    注意力汇聚器 (Attentive Pooler)

    核心思想 —— 用"查询"来提问，从"上下文"中提取答案：
        将 encoder 输出的 patch 特征视作"上下文（Key 和 Value）"，
        通过一组可学习的"查询向量（Query）"去交叉注意力地检索信息，
        最终每个查询向量得到一个汇聚后的表示。

    类比理解：
        假设你有一本很长的书（= N 个 patch 特征），你想知道这本书主要讲了什么。
        - 简单方法：把每个段落的摘要取平均（= 平均池化）
        - 本方法：你带着几个问题（= 查询向量）去读这本书，每个问题帮你从书中
          提取出相关的信息。最终你得到的是针对这些问题的答案（= 汇聚特征）。

    架构详解：

        输入 x: [Batch_Size, N_Patches, Embed_Dim]（来自 encoder 的输出）

        步骤 1: 交叉注意力 (Cross-Attention)
            Query ← query_tokens（可学习的参数，形状 [1, num_queries, Embed_Dim]）
            Key   ← x（encoder 输出的 patch 特征）
            Value ← x（encoder 输出的 patch 特征）

            标准的多头交叉注意力计算：
                Attention(Q, K, V) = softmax(Q*K^T / sqrt(d)) * V

            效果：每个查询向量关注输入序列中跟它"相关"的 patch，
                  并加权聚合这些 patch 的信息。

        步骤 2 (可选): 自注意力精炼 (Self-Attention Blocks)
            如果 depth > 1，交叉注意力之后会再接若干层自注意力。
            这些层在查询向量之间做自注意力（不涉及原始输入），
            让不同查询向量互相交流信息，进一步精炼表示。

    关键参数：
        num_queries:   查询向量的数量（通常为 1 = 一个全局表示）
        embed_dim:     特征维度（通常为 768 或 1024）
        num_heads:     多头注意力的头数
        mlp_ratio:     MLP 隐藏层维度是 embed_dim 的几倍
        depth:         交叉注意力后的自注意力层数（depth=1 表示仅交叉注意力）
        norm_layer:    归一化层的类型（通常为 nn.LayerNorm）
        init_std:      参数初始化的标准差（用于截断正态分布）
        qkv_bias:      注意力的 QKV 投影是否使用偏置项
        complete_block: True  → 使用 CrossAttentionBlock（交叉注意力 + MLP + 残差）
                        False → 仅使用 CrossAttention（仅交叉注意力，更轻量）
    """

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True
    ):
        super().__init__()

        # ----- 可学习的查询向量 (Query Tokens) -----
        # 形状: [1, num_queries, embed_dim]
        # 这些向量是模型的 nn.Parameter，在反向传播中被优化
        # 训练过程中，模型学习"该问什么问题"才能最好地汇聚 patch 信息
        # 前向传播时会在 batch 维度上重复以匹配当前批次大小
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        # ----- 交叉注意力层 -----
        # complete_block 决定使用哪种交叉注意力模块：
        #   True:  CrossAttentionBlock = 交叉注意力 + MLP + 残差连接 + LayerNorm
        #         结构: LN → CrossAttn → 残差 → LN → MLP → 残差
        #   False: CrossAttention      = 仅交叉注意力 + 残差连接（更轻量，
        #         适合参数量敏感的场景）
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        # ----- 可选的额外自注意力层 -----
        # 如果 depth > 1，在交叉注意力之后堆叠 (depth-1) 层标准 Transformer Block
        # 这些 Block 在查询向量之间做自注意力，不回头查看原始的 patch 输入
        # 类比：交叉注意力 = 从书里找答案，自注意力 = 不同问题之间互相讨论确认
        # nn.ModuleList 用于容纳可变数量的子模块（与 Python list 不同，
        # 它会被 PyTorch 正确追踪，从而参与参数更新）
        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,          # 不使用可学习的 QK 缩放因子
                                             # （使用默认的 1/sqrt(d_k) 缩放）
                    norm_layer=norm_layer)
                for i in range(depth-1)])     # depth-1 是因为第一层是交叉注意力

        # ----- 权重初始化 -----
        self.init_std = init_std
        # trunc_normal_：截断正态分布初始化
        # 从标准正态分布中采样，但超过 2 个标准差的样本会被丢弃并重新采样
        # 这样做的好处：避免出现极端值，让训练初期更稳定
        trunc_normal_(self.query_tokens, std=self.init_std)
        # self.apply() 会递归地对模块树中的每个子模块调用 _init_weights
        # 这是 PyTorch 推荐的模块范围内统一初始化方式
        self.apply(self._init_weights)
        # 对特定层的输出投影权重进行深度缩放，确保深层梯度方差稳定
        self._rescale_blocks()

    def _rescale_blocks(self):
        """
        按层深度对残差路径的输出投影权重进行缩放。

        为什么要缩放？
            在深层 Transformer 中，每一层的输出都会加到残差路径上。
            如果没有缩放，更深的层（layer_id 大）的梯度方差会与浅层不同，
            导致训练不稳定。使用 1/sqrt(2*layer_id) 缩放是 ViT 等
            工作的标准做法，可以保持各层梯度的方差一致。

        缩放公式：
            param = param / sqrt(2 * layer_id)

        缩放对象：
            - 交叉注意力层的输出投影权重（layer_id=1）
            - MLP 的第二层全连接权重（fc2，即输出投影）
            - 每个额外自注意力层的注意力输出投影和 MLP fc2

        注意：
            param.div_() 是 PyTorch 的就地除法操作（in-place），
            末尾的下划线表示直接修改张量本身，不创建副本。
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        # 交叉注意力层的投影权重：layer_id=1 表示最浅层
        if self.complete_block:
            # CrossAttentionBlock 的注意力输出投影
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            # CrossAttentionBlock 的 MLP 第二层（fc2 是输出投影）
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            # CrossAttention（简化版）的注意力输出投影
            rescale(self.cross_attention_block.proj.weight.data, 1)

        # 额外自注意力层的投影权重
        # enumerate(self.blocks, 1) 从 1 开始编号
        # 第一个自注意力层 layer_id = 2（因为交叉注意力是 1）
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """
        递归初始化所有子模块的权重。

        PyTorch 中，调用 module.apply(fn) 会按深度优先顺序遍历
        模块树中的每个子模块，并对每个子模块调用 fn(m)。
        这是实现模块范围统一初始化的标准方式。

        初始化策略：
            - nn.Linear:    权重用截断正态分布（避免极端值），偏置置零
            - nn.LayerNorm: 偏置置零、权重置 1（归一化层的"恒等映射"初始态）
            - nn.Conv2d:    同 nn.Linear（如果模块中包含了卷积层）

        参数：
            m: 模型树中的一个子模块（由 self.apply() 自动传入）
        """
        if isinstance(m, nn.Linear):
            # 全连接层的权重初始化
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # bias 初始化为常数 0：从零开始学习偏置
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm 初始化为"恒等映射"状态
            # bias=0, weight=1 意味着归一化后不做任何额外的缩放和偏移
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 卷积层（如果存在）：初始化方式与全连接层相同
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播：将 patch 特征序列汇聚成查询表示。

        参数：
            x: encoder 输出的 patch 特征
               形状: [Batch_Size, N_Patches, Embed_Dim]
               其中 N_Patches 是视频被切分后的 patch 数量

        返回值：
            q: 汇聚后的查询表示
               形状: [Batch_Size, num_queries, Embed_Dim]
               如果 num_queries=1，这通常就是最终的"视频级表示"

        计算流程：
            输入 x (B, N, D)
                      │
            query_tokens (1, Q, D)
                      │
                      ▼  repeat(B 次)
                   q (B, Q, D)
                      │
                      ├────────── x (B, N, D)
                      │            │
                      ▼            ▼
              交叉注意力 CrossAttn(Q=q, K=x, V=x)
              查询向量 q 加权"关注"所有 patch 特征
                      │
                      ▼
                   q (B, Q, D)
                      │
                      ▼  (可选: 自注意力 × (depth-1) 层)
               Block 内部: LN → SelfAttn → 残差 → LN → MLP → 残差
               q 内部的查询向量互相交流精炼
                      │
                      ▼
              最终输出 q (B, Q, D)
        """
        # 步骤 0: 在 batch 维度上重复查询向量
        # query_tokens 形状: [1, num_queries, D]
        # len(x) 获取 batch size，repeat 后形状: [B, num_queries, D]
        # 每个样本拥有一份独立的查询向量副本
        q = self.query_tokens.repeat(len(x), 1, 1)

        # 步骤 1: 交叉注意力 —— 核心操作
        # q 作为 Query（查询方），x 同时作为 Key 和 Value（信息来源）
        # 查询向量"看向" encoder 输出的所有 patch 特征，
        # 根据注意力分数加权聚合信息
        q = self.cross_attention_block(q, x)

        # 步骤 2: 可选的额外自注意力精炼
        # 每个 Block 是一个标准 Transformer 编码器层：
        # 包含多头自注意力 + MLP，每部分前后都有 LayerNorm 和残差连接
        # 此时 q 不再"看" x，只在 q 内部做自注意力交流
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)

        return q


class AttentiveClassifier(nn.Module):
    """
    注意力分类器 (Attentive Classifier)

    这是 V-JEPA 用于下游分类评估的标准分类头。
    它将 AttentivePooler 和一个线性分类层组合在一起，
    封装成一个端到端的分类器模块。

    设计理念：
        在 V-JEPA 的自监督预训练完成后，我们需要评估学到的视觉表征好不好。
        方案是冻结编码器（不更新其参数），只训练一个轻量级的分类头。
        如果用一个简单的线性分类头就能取得好的分类准确率，
        说明预训练的视觉表征本身就包含了足够好的语义信息。

    架构流程：
        encoder output (B, N, D)
              │
              ▼
        AttentivePooler       ← 核心：自适应地汇聚 patch 特征
              │
              ▼
        (B, 1, D)             ← 1 个查询向量，D 维特征
              │
              ▼
        .squeeze(1)           ← 去掉查询维度（第 1 维大小为 1），变成 (B, D)
              │
              ▼
        nn.Linear(D, C)       ← 线性分类头，C = num_classes
              │
              ▼
        (B, C)                ← 每个类别的原始分数（logits，未做 softmax）

    使用场景：
        1. 冻结评估 (Frozen Evaluation)：
           编码器参数冻结，只训练分类头。评估预训练表征的线性可分性。
        2. 微调评估 (Fine-tuning)：
           编码器和分类头一起训练。评估表征的可迁移性。

    关键参数：
        embed_dim:       特征维度（需与 encoder 输出维度一致，通常 768）
        num_heads:       注意力头数
        mlp_ratio:       MLP 隐藏层扩展比例（如 4.0 表示隐藏层 = 4 * embed_dim）
        depth:           AttentivePooler 的深度（交叉注意力 + 后续自注意力层数）
        norm_layer:      归一化层类型（nn.LayerNorm）
        init_std:        截断正态分布初始化的标准差
        qkv_bias:        注意力的 QKV 投影是否使用偏置
        num_classes:     分类任务的总类别数（如 Kinetics-400 视频数据集则设为 400）
        complete_block:  True=使用完整的 CrossAttentionBlock，False=仅用 CrossAttention
    """

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
    ):
        super().__init__()

        # 内部注意力汇聚器：将 N 个 patch 特征聚合成 1 个全局特征向量
        # num_queries=1 是固定的 —— 分类任务只需要一个视频级表示
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )

        # 线性分类层
        # 输入维度: embed_dim（池化后的特征维度）
        # 输出维度: num_classes（每个类别一个原始分数）
        # bias=True：每个类别有一个可学习的偏置项
        # 训练时通常配合 nn.CrossEntropyLoss 使用，它会自动对 logits 做 softmax
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        """
        前向传播：从 patch 特征到分类分数。

        参数：
            x: encoder 输出的 patch 特征
               形状: [Batch_Size, N_Patches, Embed_Dim]

        返回值：
            分类 logits，形状: [Batch_Size, num_classes]
            （logits 是未经 softmax 的原始分类分数，
             通常在损失函数（如 CrossEntropyLoss）内部做 softmax）

        计算流程：
            1. self.pooler(x)    → (B, 1, D)   # 注意力汇聚，得到 1 个全局特征
            2. .squeeze(1)       → (B, D)       # 去掉大小为 1 的查询维度
            3. self.linear(x)    → (B, C)       # 线性投影到类别空间
        """
        # self.pooler(x) 输出形状: [B, 1, D]（1 个 query token 的结果）
        x = self.pooler(x)
        # squeeze(1): 沿着第 1 维（索引从 0 开始）挤压
        # 因为 num_queries=1，所以第 1 维大小为 1
        # squeeze 将该维度移除，形状从 [B, 1, D] 变为 [B, D]
        x = x.squeeze(1)
        # 线性分类层: [B, D] → [B, num_classes]
        # 这个映射矩阵的大小为 D × num_classes，是分类头中主要的可训练参数
        x = self.linear(x)
        return x
