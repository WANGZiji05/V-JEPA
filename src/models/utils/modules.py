# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 基础神经网络模块
# ============================================================================
# 本文件定义了构成 Transformer 的基本"积木块"：
#   - MLP: 多层感知机（全连接网络）
#   - Attention: 多头自注意力机制（Transformer的核心）
#   - CrossAttention: 交叉注意力（query来自一处，key/value来自另一处）
#   - Block: 一个完整的Transformer块 = Attention + MLP + 残差连接
#   - CrossAttentionBlock: 交叉注意力块 = CrossAttention + MLP + 残差连接

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    多层感知机（Multi-Layer Perceptron）

    结构：Linear → GELU激活 → Dropout → Linear → Dropout
    这是Transformer中每个Block的"前馈网络"部分。
    hidden_features 通常是 in_features 的4倍（mlp_ratio=4）。

    为什么需要MLP？注意力负责"交流"（token之间交换信息），
    MLP负责"思考"（对每个token单独进行非线性变换）。
    """
    def __init__(
        self,
        in_features,              # 输入维度
        hidden_features=None,     # 隐藏层维度（默认=输入维度）
        out_features=None,        # 输出维度（默认=输入维度）
        act_layer=nn.GELU,        # 激活函数（GELU比ReLU更平滑，Transformer中常用）
        drop=0.                   # Dropout比率
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)   # 第一层：升维
        self.act = act_layer()                                # 非线性激活
        self.fc2 = nn.Linear(hidden_features, out_features)   # 第二层：降维
        self.drop = nn.Dropout(drop)                          # Dropout防止过拟合

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    多头自注意力机制（Multi-Head Self-Attention）

    这是Transformer的核心！简单来说：
    每个token会"关注"所有其他token，计算一个加权和。

    具体步骤（以单头为例）：
    1. 对输入x做三个线性投影：Query (Q), Key (K), Value (V)
       - Q: "我在找什么？"  (查询)
       - K: "我有什么？"    (键)
       - V: "我的内容是什么？" (值)
    2. 计算注意力分数: score = Q × K^T / √d
       (除以√d是为了防止点积过大导致softmax梯度消失)
    3. 用softmax归一化: attention_weights = softmax(score)
    4. 加权求和: output = attention_weights × V

    多头注意力：把特征维度分成多个"头"，每个头独立做注意力，
    最后把所有头的结果拼接起来。这样模型可以从不同角度关注信息。

    参数:
        dim: 输入/输出特征维度
        num_heads: 注意力头数
        qkv_bias: QKV投影是否使用偏置
        qk_scale: QK点积的缩放因子
        attn_drop: 注意力权重的dropout
        proj_drop: 输出投影的dropout
        use_sdpa: 是否使用PyTorch的scaled_dot_product_attention（更快）
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True  # SDPA = Scaled Dot Product Attention，PyTorch优化的注意力实现
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的特征维度
        # 如果没指定缩放因子，默认用 1/√(head_dim)
        self.scale = qk_scale or head_dim ** -0.5

        # QKV联合投影：一次性投影出Q、K、V，然后split成三份
        # 输入: [B, N, dim] → 输出: [B, N, 3*dim]
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 多头结果拼接后的线性投影
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa  # 是否使用PyTorch优化的SDPA

    def forward(self, x, mask=None):
        """
        参数:
            x: 输入 [B, N, C] (批大小, token数, 特征维度)
            mask: 可选的mask，控制哪些token可以互相注意
        """
        B, N, C = x.shape

        # 步骤1：QKV投影，并reshape成多头格式
        # [B, N, 3*dim] → [B, N, 3, num_heads, head_dim] → [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: [B, num_heads, N, head_dim]

        if self.use_sdpa:
            # PyTorch优化的注意力计算（推荐，更快更省内存）
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
                attn = None  # SDPA不返回注意力矩阵
        else:
            # 手动实现（可以看到注意力矩阵，便于调试和可视化）
            # 步骤2：计算注意力分数 Q×K^T
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            # 步骤3：softmax归一化
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            # 步骤4：加权求和
            x = (attn @ v)

        # 将多头结果合并回原始维度
        # [B, num_heads, N, head_dim] → [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """
    Transformer 基础块

    结构（标准的 Pre-Norm Transformer）：
        x = x + Attention(LayerNorm(x))    ← 自注意力 + 残差连接
        x = x + MLP(LayerNorm(x))          ← MLP + 残差连接

    "Pre-Norm" 指在注意力/MLP之前做归一化（而非之后），
    这对于训练深层网络更稳定。

    残差连接（Residual Connection）的作用：
    让梯度可以直接流过加法操作，避免深层网络中的梯度消失问题。
    """
    def __init__(
        self,
        dim,                     # 特征维度
        num_heads,               # 注意力头数
        mlp_ratio=4.,            # MLP隐藏层倍数（hidden_dim = dim * mlp_ratio）
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        act_layer=nn.GELU,       # 激活函数
        norm_layer=nn.LayerNorm, # 归一化层
        grid_size=None,          # （非标准ViT参数，可忽略）
        grid_depth=None,         # （非标准ViT参数，可忽略）
    ):
        super().__init__()
        # 第一个归一化层 + 自注意力
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        # 第二个归一化层 + MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP隐藏层维度
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        """
        参数:
            x: 输入 [B, N, C]
            return_attention: 是否返回注意力矩阵（用于可视化）
            mask: 可选的注意力mask
        """
        # Pre-Norm + 自注意力 + 残差
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y  # 残差连接
        # Pre-Norm + MLP + 残差
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """
    交叉注意力（Cross-Attention）

    与自注意力的区别：
    - 自注意力: Q、K、V都来自同一个输入 (自己关注自己)
    - 交叉注意力: Q来自一个输入，K和V来自另一个输入

    在V-JEPA中的用途：
    Attentive Pooler使用交叉注意力，让一个可学习的query token
    去"查询"编码器输出的所有token，从而聚合出全局特征。

    参数:
        dim: 特征维度
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        use_sdpa: 是否使用PyTorch SDPA
    """
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)       # Q投影（从query来）
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)  # KV联合投影（从context来）
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        """
        参数:
            q: query token [B, n, C]  (n通常很小，如1)
            x: context token [B, N, C] (N是所有patch的token数)
        """
        B, n, C = q.shape
        # Q投影：[B, n, C] → [B, num_heads, n, head_dim]
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        # KV投影：[B, N, C] → [B, N, 2*C] → [2, B, num_heads, N, head_dim]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, num_heads, N, head_dim]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            # 手动计算交叉注意力
            xattn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, n, N]
            xattn = xattn.softmax(dim=-1)
            q = (xattn @ v)

        # [B, num_heads, n, head_dim] → [B, n, C]
        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块 = CrossAttention + MLP + 残差连接

    类似于标准的Transformer Block，但把自注意力换成了交叉注意力。

    结构：
        q = q + CrossAttention(q, LayerNorm(x))  ← 用q查询x，残差连接
        q = q + MLP(LayerNorm(q))                 ← MLP处理，残差连接

    在V-JEPA评估中的用途：
    Attentive Classifier的核心 = CrossAttentionBlock + Linear
    用可学习的query去聚合视频/图像特征，然后分类。
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        # q查询x（交叉注意力）+ 残差
        y = self.xattn(q, self.norm1(x))
        q = q + y
        # MLP + 残差
        q = q + self.mlp(self.norm2(q))
        return q
