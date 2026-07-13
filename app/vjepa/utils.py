# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 训练工具函数
# ============================================================================
# 包含：
# - load_checkpoint: 从磁盘加载训练checkpoint（模型参数+优化器状态）
# - init_video_model: 构造V-JEPA的三组件（编码器+预测器+MultiMask包装）
# - init_opt: 初始化优化器(AdamW)+学习率调度器+权重衰减调度器

import logging
import sys
import warnings
import yaml

import torch

import src.models.vision_transformer as video_vit    # ViT编码器
import src.models.predictor as vit_pred              # ViT预测器
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,   # 带warmup的余弦学习率衰减
    CosineWDSchedule)       # 余弦权重衰减调度
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    r_path,          # checkpoint文件路径
    encoder,         # 编码器模型
    predictor,       # 预测器模型
    target_encoder,  # 目标编码器模型
    opt,             # 优化器
    scaler,          # 混合精度scaler
):
    """
    从磁盘加载训练checkpoint

    checkpoint包含：
    - encoder/predictor/target_encoder的参数
    - 优化器状态（动量、方差等）
    - 混合精度scaler状态
    - 训练到的epoch数
    """
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')

    epoch = 0
    try:
        epoch = checkpoint['epoch']

        # 加载编码器参数
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # 加载预测器参数
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # 加载目标编码器参数
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}'
            )

        # 加载优化器和scaler状态
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',        # 模型规模：vit_tiny/vit_small/vit_base/vit_large/vit_huge
    crop_size=224,
    pred_depth=6,                 # 预测器深度（层数）
    pred_embed_dim=384,           # 预测器内部维度
    uniform_power=False,          # 是否均匀分配3D位置编码频率
    use_mask_tokens=False,        # 是否使用可学习mask token
    num_mask_tokens=2,            # mask token种类数
    zero_init_mask_tokens=True,   # 是否零初始化mask token
    use_sdpa=False,               # 是否使用PyTorch的SDPA
):
    """
    初始化V-JEPA模型的三组件

    返回:
        encoder: MultiMaskWrapper包装的ViT编码器
        predictor: PredictorMultiMaskWrapper包装的ViT预测器

    注意：编码器和预测器都被MultiMask包装器包裹，
    这使得它们可以同时处理多种mask策略。
    """

    # 1. 创建编码器
    # model_name如'vit_huge'通过字典查找调用对应的构造函数
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    # 用MultiMask包装器包裹（支持多种mask策略）
    encoder = MultiMaskWrapper(encoder)

    # 2. 创建预测器
    # 预测器比编码器小得多（维度更低，层数更少）
    predictor = vit_pred.__dict__['vit_predictor'](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,  # 继承编码器的维度
        predictor_embed_dim=pred_embed_dim,     # 预测器内部使用更小的维度
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,   # 注意力头数与编码器相同
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
    )
    # 用PredictorMultiMask包装器包裹
    predictor = PredictorMultiMaskWrapper(predictor)

    # 3. 权重初始化（对模型所有模块再次初始化以确保一致性）
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    # 4. 将模型移到GPU
    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    # 5. 打印可训练参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,  # 每个epoch的迭代次数
    start_lr,              # warmup起始学习率
    ref_lr,                # 参考/基础学习率
    warmup,                # warmup的epoch数
    num_epochs,            # 总epoch数
    wd=1e-6,               # 初始权重衰减
    final_wd=1e-6,         # 最终权重衰减
    final_lr=0.0,          # 最终学习率
    mixed_precision=False,  # 是否使用混合精度
    ipe_scale=1.25,        # epoch长度缩放因子
    betas=(0.9, 0.999),    # AdamW的beta参数
    eps=1e-8,              # AdamW的epsilon
    zero_init_bias_wd=True,  # 是否对bias参数禁用权重衰减
):
    """
    初始化优化器和学习率调度器

    V-JEPA使用AdamW优化器 + 余弦学习率衰减 + 余弦权重衰减。
    学习率曲线：warmup → 保持 → 余弦衰减 → final_lr
    权重衰减曲线：初始值 → 余弦增长到最终值

    参数分组策略（不同的参数用不同的优化设置）：
    1. 权重参数（encoder+pred）：使用权重衰减
    2. bias和1D参数：不使用权重衰减
    这种分组策略是ViT训练的标准做法。
    """
    # 参数分组
    param_groups = [
        # 组1: encoder的权重参数（bias和1D参数除外）
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        },
        # 组2: predictor的权重参数
        {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        },
        # 组3: encoder的bias和1D参数（不禁用权重衰减）
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        },
        # 组4: predictor的bias和1D参数
        {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        },
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    # 余弦学习率调度（带warmup）
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    # 余弦权重衰减调度
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    # 混合精度scaler（只有混合精度训练时才需要）
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
