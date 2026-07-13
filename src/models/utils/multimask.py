# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# MultiMask 包装器 —— 支持多种mask策略并行处理
# ============================================================================
# V-JEPA 的核心创新之一是使用多种不同的mask策略同时训练。
# mask策略定义了哪些视频区域是"上下文"(context，编码器能看到的部分)，
# 哪些是"目标"(target，需要预测的部分)。
#
# 为什么需要多种mask？
# 不同的mask迫使模型从不同角度理解视频：
#   - 小块mask（短时+小空间）：学习局部细节和短时动态
#   - 大块mask（长时+大空间）：学习全局语义和长时依赖
#
# MultiMaskWrapper的工作方式：
# 对于同一个输入，依次用每种mask策略调用backbone，
# 然后将所有结果收集成列表返回。
#
# 例如：有2种mask策略 → backbone被调用2次 → 返回2个输出组成的列表

import torch.nn as nn


class MultiMaskWrapper(nn.Module):
    """
    编码器的 MultiMask 包装器

    作用：让同一个编码器可以处理多种mask策略。
    对于输入x，依次用每个mask调用backbone，返回所有结果的列表。

    使用方式：
        encoder = MultiMaskWrapper(vision_transformer)
        # masks = [mask1, mask2]  # 两个不同的mask策略
        outputs = encoder(x, masks=masks)
        # outputs = [output_for_mask1, output_for_mask2]

    这在V-JEPA训练中很重要：一个视频样本会被多种mask策略处理，
    每种mask产生的context/target对都会被用来计算损失。
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # 被包装的编码器（如VisionTransformer）

    def forward(self, x, masks=None):
        # 没有mask → 直接返回backbone输出（评估时使用）
        if masks is None:
            return self.backbone(x)

        # 确保masks是列表（单个mask也包装成列表）
        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]

        # 对每种mask策略分别调用backbone
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs  # 返回列表，每个元素对应一种mask策略的输出


class PredictorMultiMaskWrapper(nn.Module):
    """
    预测器的 MultiMask 包装器

    与编码器包装器类似，但预测器需要更多输入：
    - ctxt: 上下文token（编码器输出）
    - tgt: 目标token（目标编码器输出）
    - masks_ctxt: 上下文区域的mask索引
    - masks_tgt: 目标区域的mask索引
    - mask_index: 告诉预测器使用哪个mask token（当使用可学习mask token时）

    由于编码器已经按每种mask策略产生了各自的上下文输出(ctxt列表)，
    预测器需要将对应的ctxt, tgt, mask_ctxt, mask_tgt配对处理。
    每种mask策略都对应一个独立的预测任务。
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # 被包装的预测器

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        """
        参数:
            ctxt: 上下文token列表（编码器对每种mask策略的输出）
            tgt: 目标token列表（目标编码器对每种mask策略的输出）
            masks_ctxt: 上下文mask索引列表
            masks_tgt: 目标mask索引列表

        返回:
            outs: 预测器对每种mask策略的预测结果列表
        """
        # 统一格式为列表
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        # 将每种mask策略的对应输入配对处理
        # zip配对: (ctxt[0], tgt[0], mask_ctxt[0], mask_tgt[0]), (ctxt[1], ...), ...
        # mask_index=i 用于选择对应的可学习mask token
        outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs
