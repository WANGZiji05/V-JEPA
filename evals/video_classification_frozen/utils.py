# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 视频评估工具模块
# ============================================================================
#
# 本模块提供了视频分类评估所需的核心工具类和数据变换。
#
# 【核心组件】
#
# 1. FrameAggregation —— 逐帧聚合策略
#    - 适用场景：预训练模型是逐帧训练的（pretrain_frames_per_clip == 1）
#    - 工作原理：将视频中每一帧独立送入编码器，然后将所有帧的 token 拼接起来
#    - 类比：就像一个静态的图像编码器，先对每帧提取特征，再把所有帧拼成一段"长文本"
#
# 2. ClipAggregation —— 逐片段聚合策略
#    - 适用场景：预训练模型是以 clip 为单位训练的（pretrain_frames_per_clip > 1）
#    - 工作原理：将每个时空 clip 独立送入编码器，然后拼接各 clip 的 token
#    - 类比：每个 clip 是视频中的一个"短语"，编码器理解每个短语，然后拼接起来
#    - 支持 attend_across_segments：让分类器跨片段交互（类似读完整段话后理解含义）
#
# 3. make_transforms —— 数据增强工厂函数
#    - 根据训练/验证模式和多视角设置选择合适的变换策略
#
# 4. VideoTransform —— 训练时的数据增强变换
#    - 包含随机裁剪缩放、自动增强、随机擦除等
#
# 5. EvalVideoTransform —— 验证时的多视角采样变换
#    - 从视频帧中裁剪多个空间视角，用于更准确的评估
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 视频数据增强模块
import src.datasets.utils.video.transforms as video_transforms     # 视频变换（Resize, Crop等）
import src.datasets.utils.video.volume_transforms as volume_transforms  # 体积变换（ClipToTensor等）
from src.datasets.utils.video.randerase import RandomErasing       # 随机擦除增强

# 位置编码工具
from src.models.utils.pos_embs import get_1d_sincos_pos_embed      # 生成 1D 正弦余弦位置编码
from src.masks.utils import apply_masks                             # 将 mask 应用于位置编码


class FrameAggregation(nn.Module):
    """
    逐帧聚合：将视频的每一帧独立编码，然后拼接所有帧的特征。

    【工作原理】
    假设一个视频有 T 帧，我们希望得到它的整体特征表示。
    1. 将 [B, C, T, H, W] 变形为 [B*T, C, H, W]（每一帧当作独立图像）
    2. 将所有帧一起送入 ViT 编码器（并行处理，效率高）
    3. 编码器输出 [B*T, N, D]（N 个空间 token，D 维特征）
    4. 变形回 [B, T*N, D]（将所有帧的 token 拼接在一起）

    【使用场景】
    当 V-JEPA 预训练时 frames_per_clip=1（即逐帧训练），评估时用此策略。
    因为模型的 patch 嵌入层不包含时间维度，所以每帧独立处理是正确的。

    【参数说明】
        model: V-JEPA Vision Transformer 编码器
        max_frames: 支持的最大帧数（用于位置编码）
        use_pos_embed: 是否添加时间位置编码
        attend_across_segments: 是否跨片段交互（本类目前未完全实现 False 分支）
    """
    def __init__(
        self,
        model,
        max_frames=10000,
        use_pos_embed=False,
        attend_across_segments=False
    ):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim = model.embed_dim   # 特征维度
        self.num_heads = model.num_heads                # 注意力头数
        self.attend_across_segments = attend_across_segments

        # 可选的时间位置编码
        # 位置编码告诉模型每一帧在时间序列中的位置
        # 用正弦余弦函数生成，不需要训练
        self.pos_embed = None
        if use_pos_embed:
            # max_frames 是支持的最大帧数
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_frames, embed_dim),
                requires_grad=False)   # 不可训练
            # 用预计算的正弦余弦位置编码初始化
            sincos = get_1d_sincos_pos_embed(embed_dim, max_frames)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        """
        前向传播。

        参数:
            x: 视频数据，结构为 List[temporal_view][spatial_view]
               每个元素是 Tensor[B, C, T, H, W]
               - B: batch size
               - C: 通道数 (3 for RGB)
               - T: 帧数
               - H, W: 空间分辨率
            clip_indices: 片段索引（用于位置编码）

        返回:
            List[Tensor]: 每个空间视角对应的特征 [B, T*N, D]
        """
        # 当前 TODO: attend_across_segments=False 的分支尚未实现
        # num_clips = len(x)
        num_views_per_clip = len(x[0])  # 每个时间片段的空间视角数

        # ---- 步骤 1: 沿 batch 维度拼接空间视角 ----
        # 例如：有 2 个空间视角，每个 [B, C, T, H, W]
        # torch.cat(xi, dim=0) → [2*B, C, T, H, W]
        x = [torch.cat(xi, dim=0) for xi in x]

        # ---- 步骤 2: 沿时间维度拼接时间片段 ----
        # 将所有时间片段按帧拼接：dim=2 是时间 C 维度
        x = torch.cat(x, dim=2)
        B, C, T, H, W = x.size()

        # ---- 步骤 3: 将帧沿 batch 维度展开 ----
        # permute(0, 2, 1, 3, 4): [B, C, T, H, W] → [B, T, C, H, W]
        # reshape(B*T, C, H, W): 每条数据变成 [C, H, W]（标准图像格式）
        # 此时每帧是独立的"图像"，可以送入标准的 ViT 编码器
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        # ---- 步骤 4: 编码器前向传播 ----
        # 输出 [B*T, N, D]
        # N: 每帧的 patch token 数（例如 224/16=14, 14*14=196 tokens + 1 CLS = 197）
        # D: 特征维度
        outputs = self.model(x)
        _, N, D = outputs.size()

        # ---- 步骤 5: 变回原始结构并展平 ----
        # [B*T, N, D] → [B, T, N, D] → [B, T*N, D]
        # 将所有帧的 token 拼接在一起
        outputs = outputs.reshape(B, T, N, D).flatten(1, 2)

        # ---- 步骤 6: 按空间视角分离 ----
        # 把它们恢复为独立的列表项
        B = B // num_views_per_clip
        all_outputs = []
        for i in range(num_views_per_clip):
            o = outputs[i*B:(i+1)*B]

            # 可选：添加时间位置编码
            if (self.pos_embed is not None) and (clip_indices is not None):
                pos_embed = self.pos_embed.repeat(B, 1, 1)        # [B, max_F, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)  # 按索引选择
                pos_embed = torch.cat(pos_embed, dim=1)           # 拼接所有时间片段
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T_total, N, D]
                pos_embed = pos_embed.flatten(1, 2)               # [B, T_total*N, D]
                o += pos_embed   # 位置编码加法（与 Transformer 的标准做法一致）
            all_outputs += [o]

        return all_outputs


class ClipAggregation(nn.Module):
    """
    逐片段聚合：将视频的每个时空片段独立编码，然后拼接所有片段的特征。

    【工作原理】
    与 FrameAggregation 不同，ClipAggregation 利用模型的时间处理能力。
    假设视频被分为 K 个片段，每个片段 T 帧：
    1. 每个 [B, C, T, H, W] clip 独立送入 ViT 编码器
       编码器内部通过 tubelet 机制处理时间维度的信息
    2. 编码器输出 [B, N*T_t, D]
       T_t = T / tubelet_size（时间 token 数）
    3. 将所有片段的时间 token 拼接 → [B, K*N*T_t, D]

    【tubelet 机制解释】
    tubelet_size 将时间维度也纳入 patch 嵌入范围。
    例如 tubelet_size=2, patch_size=16：
    输入 [C, 16, 224, 224] → 提取 16×16×2 的时空 patch
    这允许模型学习短时间的运动模式。

    【attend_across_segments 参数】
    - False: 每个 (空间视角, 时间片段) 组合被独立分类
             最终通过平均 softmax 概率聚合
    - True:  所有时间片段的 token 被拼接成一个长序列
             分类器通过一次注意力操作处理所有片段（允许跨片段交互）

    【参数说明】
        model: V-JEPA Vision Transformer 编码器
        tubelet_size: 时间维度 patch 大小
        max_frames: 最大支持帧数
        use_pos_embed: 是否使用时间位置编码
        attend_across_segments: 是否让分类器跨片段交互
    """
    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=10000,
        use_pos_embed=False,
        attend_across_segments=False
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments

        # 时间位置编码
        # 注意：max_T 需要考虑 tubelet 的压缩效果
        # 例如 max_frames=10000, tubelet_size=2 → max_T=5000
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_T, embed_dim),
                requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        """
        前向传播。

        参数:
            x: List[temporal_view][spatial_view]，每个元素是 Tensor[B, C, T, H, W]
            clip_indices: 片段索引，List[Tensor[B, num_frames]]

        当 attend_across_segments=False 时返回:
            List[List[Tensor]]: [空间视角][时间片段] 的特征
                               形状为 [B, N*T_t, D]

        当 attend_across_segments=True 时返回:
            List[Tensor]: [空间视角] 所有时间片段拼接后的特征
                          形状为 [B, num_clips*N*T_t, D]
        """

        num_clips = len(x)              # 时间片段数
        num_views_per_clip = len(x[0])   # 每个片段的空间视角数
        B, C, T, H, W = x[0][0].size()  # 获取数据形状

        # ---- 步骤 1: 将所有 clip 沿 batch 维度拼接 ----
        # 先将每个片段的空间视角在 batch 维拼接
        # [B,C,T,H,W]*spatial_views → [B*spatial_views, C, T, H, W]
        x = [torch.cat(xi, dim=0) for xi in x]
        # 再将所有片段在 batch 维拼接
        # → [num_clips * B * spatial_views, C, T, H, W]
        x = torch.cat(x, dim=0)

        # ---- 步骤 2: 编码器前向传播 ----
        # 利用 tubelet 机制，编码器同时处理空间和时间信息
        # 输出 [total_B, N*T_t, D]
        outputs = self.model(x)
        _, N, D = outputs.size()

        # 计算 token 数量
        # T_t: 时间 token 数（经过 tubelet 压缩后的帧数）
        T = T // self.tubelet_size  # 每个 clip 的时间 token 数
        N = N // T                   # 空间 token 数
        # 例如 T=16, tubelet_size=2 → T_t=8 个时间 token
        # N=197 (14x14 + CLS), N_spatial=197/8 → 实际上 = 整理后 14x14 个空间位置有所不同

        # ---- 步骤 3: 按 [空间视角, 时间片段] 解包输出 ----
        # eff_B: 每个时间片段的有效 batch size（= B * num_views_per_clip）
        eff_B = B * num_views_per_clip

        # all_outputs[j] 包含所有时间片段的第 j 个空间视角
        all_outputs = [[] for _ in range(num_views_per_clip)]
        for i in range(num_clips):                          # 遍历时间片段
            o = outputs[i*eff_B:(i+1)*eff_B]                # 取出当前片段
            for j in range(num_views_per_clip):              # 遍历空间视角
                all_outputs[j].append(o[j*B:(j+1)*B])       # 分配到对应列表

        # ---- 如果不跨片段交互，直接返回 ----
        if not self.attend_across_segments:
            return all_outputs   # List[List[Tensor]] 结构

        # ---- 如果跨片段交互，拼接所有时间片段 ----
        for i, outputs in enumerate(all_outputs):
            # 每个 o: [B, N, D]
            # 其中 N = T_t * N_spatial（时间 token * 空间 token）
            # 变形为 [B, T_t, N_spatial, D]
            outputs = [o.reshape(B, T, N, D) for o in outputs]
            # 在时间维度拼接 → [B, num_clips*T_t, N_spatial, D]
            # 然后展平 → [B, num_clips*T_t*N_spatial, D]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)

            # 可选：添加时间位置编码
            if (self.pos_embed is not None) and (clip_indices is not None):
                # 考虑 tubelet 对索引的影响
                clip_indices = [c[:, ::self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)    # [B, max_T, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)
                pos_embed = torch.cat(pos_embed, dim=1)        # [B, total_T, D]
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, total_T, N, D]
                pos_embed = pos_embed.flatten(1, 2)            # [B, total_T*N, D]
                outputs += pos_embed

            all_outputs[i] = outputs

        return all_outputs


def make_transforms(
    training=True,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    num_views_per_clip=1,
    normalize=((0.485, 0.456, 0.406),      # ImageNet 均值
               (0.229, 0.224, 0.225))       # ImageNet 标准差
):
    """
    数据增强工厂函数：根据配置选择合适的视频变换。

    验证模式 + 多视角 → EvalVideoTransform（多空间视角裁剪）
    否则 → VideoTransform（标准增强）

    参数:
        training (bool): 训练/验证模式
        random_horizontal_flip (bool): 随机水平翻转
        random_resize_aspect_ratio (tuple): 随机缩放的宽高比范围
        random_resize_scale (tuple): 随机缩放的尺度范围
        reprob (float): 随机擦除概率
        auto_augment (bool): 是否使用自动增强
        motion_shift (bool): 是否使用运动偏移增强
        crop_size (int): 裁剪尺寸
        num_views_per_clip (int): 空间视角数
        normalize (tuple): 归一化参数

    返回:
        VideoTransform 或 EvalVideoTransform 对象
    """

    if not training and num_views_per_clip > 1:
        # 验证时的多视角模式：从多个空间位置裁剪
        print('Making EvalVideoTransform, multi-view')
        _frames_augmentation = EvalVideoTransform(
            num_views_per_clip=num_views_per_clip,
            short_side_size=crop_size,
            normalize=normalize,
        )
    else:
        # 训练或单视角验证：标准视频变换
        _frames_augmentation = VideoTransform(
            training=training,
            random_horizontal_flip=random_horizontal_flip,
            random_resize_aspect_ratio=random_resize_aspect_ratio,
            random_resize_scale=random_resize_scale,
            reprob=reprob,
            auto_augment=auto_augment,
            motion_shift=motion_shift,
            crop_size=crop_size,
            normalize=normalize,
        )
    return _frames_augmentation


class VideoTransform(object):
    """
    视频数据变换（主要用于训练和单视角验证）。

    执行以下变换流水线：
    训练模式：
      1. 将帧转为 PIL Image（如需要）
      2. 自动增强（如启用）
      3. 转为 Tensor → [T, C, H, W]
      4. 重排为 [T, H, W, C]
      5. 归一化
      6. 重排为 [C, T, H, W]
      7. 随机裁剪缩放
      8. 随机水平翻转
      9. 随机擦除（如启用）

    验证模式：
      1. Resize 到短边 256/224*crop_size
      2. 中心裁剪
      3. 转为 Tensor 并归一化

    【参数说明】
        training: 训练模式
        random_horizontal_flip: 随机水平翻转概率
        random_resize_aspect_ratio: 随机缩放宽高比范围
        random_resize_scale: 随机缩放尺度范围
        reprob: 随机擦除概率
        auto_augment: 是否使用 RandAugment 自动增强
        motion_shift: 是否启用运动偏移增强（对视频时间维度做随机偏移）
        crop_size: 最终裁剪尺寸
        normalize: 归一化参数 (mean, std)
    """
    def __init__(
        self,
        training=True,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):

        self.training = training

        # 验证时的标准变换：缩放 + 中心裁剪 + 归一化
        # 短边缩放到 crop_size * 256/224，这是 ImageNet 评估的标准做法
        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose([
            video_transforms.Resize(short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),          # 将视频 clip 转为 tensor
            video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.normalize = torch.tensor(normalize)

        # 自动增强：RandAugment 的一种实现
        # rand-m7-n4-mstd0.5-inc1: 7 种增强操作，幅度 4，标准差 0.5
        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        # 空间变换：随机裁剪缩放
        # motion_shift=True 时使用带时间偏移的版本（不同帧有略微不同的裁剪位置）
        self.spatial_transform = video_transforms.random_resized_crop_with_shift \
            if motion_shift else video_transforms.random_resized_crop

        # 随机擦除：在图像中随机擦除一个矩形区域
        # 这是一种正则化技术，模拟遮挡场景
        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',       # 用随机像素值填充擦除区域
            max_count=1,        # 最多擦除 1 个区域
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):
        """
        对视频帧序列应用变换。

        参数:
            buffer: 视频帧列表，每个元素是 [H, W, C] 的 numpy 数组或 PIL Image

        返回:
            List[Tensor]: 变换后的视频，包裹在列表中
                         Tensor 形状为 [C, T, H, W]（通道优先）
        """

        # 验证模式：直接使用标准评估变换
        if not self.training:
            return [self.eval_transform(buffer)]

        # ---- 训练模式变换流水线 ----

        # 步骤 1: 将 numpy 数组转为 PIL Image（randaugment 需要 PIL 格式）
        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        # 步骤 2: 自动增强（RandAugment）
        if self.auto_augment:
            buffer = self.autoaug_transform(buffer)

        # 步骤 3: 转为 Tensor [T, C, H, W]
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # 沿第 0 维堆叠 → [T, C, H, W]

        # 步骤 4: 重排为 [T, H, W, C]（归一化函数需要的格式）
        buffer = buffer.permute(0, 2, 3, 1)  # C→H, H→W, W→C

        # 步骤 5: 归一化
        buffer = tensor_normalize(buffer, self.normalize[0], self.normalize[1])

        # 步骤 6: 重排为 [C, T, H, W]（空间裁剪函数需要的格式）
        buffer = buffer.permute(3, 0, 1, 2)

        # 步骤 7: 随机裁剪缩放
        # scale: 缩放范围
        # ratio: 宽高比范围
        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )

        # 步骤 8: 随机水平翻转（50% 概率）
        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        # 步骤 9: 随机擦除
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)  # [C, T, H, W] → [T, C, H, W]
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # [T, C, H, W] → [C, T, H, W]

        return [buffer]


class EvalVideoTransform(object):
    """
    视频评估时的多视角变换。

    【工作原理】
    从视频帧中裁剪多个空间视角，每个视角覆盖不同的区域。
    这对于准确的视频分类评估很重要，因为动作可能发生在画面的任何位置。

    示例：对于 256x256 的帧，crop_size=224，num_views=3
      - 视角 1: 裁剪区域 [0:224, 0:224]（左上）
      - 视角 2: 裁剪区域 [16:240, 0:224]（左中）
      - 视角 3: 裁剪区域 [32:256, 0:224]（左下）

    最终预测 = 所有视角预测的平均值

    【参数说明】
        num_views_per_clip: 每个片段的空间视角数
        short_side_size: 短边缩放尺寸
        normalize: 归一化参数
    """
    def __init__(
        self,
        num_views_per_clip=1,
        short_side_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):
        self.views_per_clip = num_views_per_clip
        self.short_side_size = short_side_size

        # 缩放变换（将短边缩放到指定尺寸，保持宽高比）
        self.spatial_resize = video_transforms.Resize(short_side_size, interpolation='bilinear')

        # 后续的转换链：转为 tensor + 归一化
        self.to_tensor = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

    def __call__(self, buffer):
        """
        对视频帧生成多个空间视角。

        参数:
            buffer: 视频帧列表或 numpy 数组，[T, H, W, C]

        返回:
            List[Tensor]: 多个空间视角的 tensor 列表
                         每个 tensor 形状为 [C, T, crop_size, crop_size]
        """

        # 缩放帧
        buffer = np.array(self.spatial_resize(buffer))
        T, H, W, C = buffer.shape

        # 计算空间视角的裁剪位置
        num_views = self.views_per_clip
        side_len = self.short_side_size

        # spatial_step: 相邻视角之间在长边方向上的偏移
        # 例如 H=256, crop=224, views=3
        # spatial_step = (max(256, 256) - 224) / (3 - 1) = 16
        spatial_step = (max(H, W) - side_len) // (num_views - 1)

        all_views = []
        for i in range(num_views):
            start = i*spatial_step
            # 沿长边方向裁剪
            if H > W:
                # 高度是长边：沿高度方向滑动裁剪
                view = buffer[:, start:start+side_len, :, :]
            else:
                # 宽度是长边：沿宽度方向滑动裁剪
                view = buffer[:, :, start:start+side_len, :]
            # 转为 tensor 并归一化
            view = self.to_tensor(view)
            all_views.append(view)

        return all_views


def tensor_normalize(tensor, mean, std):
    """
    对 tensor 进行标准化：减去均值，除以标准差。

    这是深度学习图像处理的标准预处理步骤，确保输入数据的
    每个通道都大致符合 0 均值和单位方差的分布，有利于模型训练和推理。

    参数:
        tensor (torch.Tensor): 输入张量（可以是 uint8 或 float 类型）
        mean (tensor or list): 每个通道的均值
        std (tensor or list): 每个通道的标准差

    返回:
        torch.Tensor: 标准化后的张量
    """
    # 如果是 uint8（0-255 整数像素值），先转为 float 并缩放到 [0, 1]
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0

    # 确保 mean 和 std 是 tensor 格式
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)

    # 标准化：(x - mean) / std
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
