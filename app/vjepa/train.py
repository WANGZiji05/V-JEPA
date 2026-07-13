# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 预训练主循环 —— 整个项目的核心训练文件
# ============================================================================
# 这个文件实现了 V-JEPA 的完整训练流程。
# 建议先了解 JEPA 的核心思想再读代码：
#
# 【V-JEPA 训练过程概览】
# 1. 从视频中随机采样一个clip（如16帧）
# 2. 生成mask：定义哪些区域是"上下文"(context)，哪些是"目标"(target)
# 3. 编码器(Encoder)只看到上下文区域 → 生成上下文特征
# 4. 目标编码器(Target Encoder)看到完整视频 → 生成目标特征
# 5. 预测器(Predictor)根据上下文特征 + 目标位置 → 预测目标区域的特征
# 6. Loss = |预测特征 - 目标特征|^p  (p通常=1，即L1 loss)
# 7. 只更新编码器和预测器，目标编码器通过EMA(指数移动平均)平滑更新
#
# 【关键设计决策】
# - 目标编码器不通过梯度更新（EMA更新），防止模型"坍缩"到平凡解
# - 预测在特征空间（非像素空间）进行，更高效
# - 多种mask策略同时训练，学习不同粒度的特征
# - 对目标特征添加扩散噪声，防止预测器走捷径

import os

# -- 分布式训练：确保每个进程只看到一个GPU
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from app.vjepa.transforms import make_transforms


# -- 训练相关的常量设置
log_timings = True
log_freq = 10          # 每10个iteration打印一次日志
checkpoint_freq = 1    # 每个epoch保存一次checkpoint

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True  # 自动选择最优的cuDNN算法

logger = get_logger(__name__)


def main(args, resume_preempt=False):
    """
    V-JEPA 预训练的主函数

    这个函数从配置文件中读取所有参数，然后执行完整的训练循环。
    配置文件包含了所有超参数（模型大小、学习率、数据路径等）。

    参数:
        args: 从YAML配置文件解析出的参数字典
        resume_preempt: 是否从preemption恢复训练（slurm集群功能）
    """

    # ==================================================================== #
    # 第一部分：从配置文件中解析所有超参数
    # ==================================================================== #

    # -- META（元数据设置）
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)  # 是否使用PyTorch SDPA优化
    which_dtype = cfgs_meta.get('dtype')  # 训练精度：float32/float16/bfloat16
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16       # bfloat16: 更大的动态范围，但精度低
        mixed_precision = True       # 混合精度训练（前向用低精度，反向用高精度）
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32        # 标准32位精度
        mixed_precision = False

    # -- MASK（遮罩策略参数）
    cfgs_mask = args.get('mask')  # 遮罩策略列表，可能包含多种策略

    # -- MODEL（模型架构参数）
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')        # 如 'vit_huge', 'vit_large'
    pred_depth = cfgs_model.get('pred_depth')         # 预测器深度
    pred_embed_dim = cfgs_model.get('pred_embed_dim') # 预测器内部维度
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)  # 是否用可学习mask token
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

    # -- DATA（数据相关参数）
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])     # 数据集路径列表
    datasets_weights = cfgs_data.get('datasets_weights', None)  # 各数据集采样权重
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), \
            'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')           # 每个GPU的batch大小
    num_clips = cfgs_data.get('num_clips')             # 每个视频采样几个clip
    num_frames = cfgs_data.get('num_frames')           # 每个clip的帧数
    tubelet_size = cfgs_data.get('tubelet_size')       # 时间patch大小
    sampling_rate = cfgs_data.get('sampling_rate')     # 帧采样间隔
    duration = cfgs_data.get('clip_duration', None)    # clip的时长（秒）
    crop_size = cfgs_data.get('crop_size', 224)        # 空间裁剪尺寸
    patch_size = cfgs_data.get('patch_size')           # ViT的patch大小
    pin_mem = cfgs_data.get('pin_mem', False)          # 是否使用pinned memory加速
    num_workers = cfgs_data.get('num_workers', 1)      # 数据加载的worker数
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    # -- DATA AUG（数据增强参数）
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)  # 运动偏移增强
    reprob = cfgs_data_aug.get('reprob', 0.)  # 随机擦除概率
    use_aa = cfgs_data_aug.get('auto_augment', False)  # 自动增强策略

    # -- LOSS（损失函数参数）
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')     # loss的指数（1=L1，2=L2）
    reg_coeff = cfgs_loss.get('reg_coeff')   # 正则化系数

    # -- OPTIMIZATION（优化器参数）
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)           # iterations per epoch
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)  # 梯度裁剪阈值
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')           # 学习率warmup的epoch数
    start_lr = cfgs_opt.get('start_lr')       # warmup起始学习率
    lr = cfgs_opt.get('lr')                   # 基础学习率
    final_lr = cfgs_opt.get('final_lr')       # 最终学习率
    ema = cfgs_opt.get('ema')                 # EMA动量范围 [start, end]
    betas = cfgs_opt.get('betas', (0.9, 0.999))  # AdamW的beta参数
    eps = cfgs_opt.get('eps', 1.e-8)

    # -- LOGGING（日志路径设置）
    cfgs_logging = args.get('logging')
    folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')

    # ==================================================================== #
    # 第二部分：初始化分布式训练环境和模型
    # ==================================================================== #

    # 设置随机种子（保证可复现性）
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')  # 多进程启动方式（兼容性最好）
    except Exception:
        pass

    # 初始化分布式训练（多GPU通信）
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    # rank: 当前进程的编号（0到world_size-1）
    # world_size: 总的GPU数量

    # 设置设备
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')  # 每个进程只用一张GPU
        torch.cuda.set_device(device)

    # 设置日志和checkpoint路径
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # 创建CSV日志记录器（记录训练指标）
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),       # JEPA预测loss
        ('%.5f', 'reg-loss'),         # 正则化loss
        ('%.5f', 'enc-grad-norm'),    # 编码器梯度范数
        ('%.5f', 'pred-grad-norm'),   # 预测器梯度范数
        ('%d', 'gpu-time(ms)'),
        ('%d', 'wall-time(ms)'),
    )

    # ==================================================================== #
    # 第三部分：构建 V-JEPA 的三个核心网络
    # ==================================================================== #
    # 1. encoder（编码器）：看到上下文(context)区域，生成上下文特征
    # 2. predictor（预测器）：根据上下文特征预测目标区域的特征
    # 3. target_encoder（目标编码器）：看到完整视频，生成目标特征（训练目标）
    # 注意：target_encoder不通过梯度更新，而是通过EMA从encoder更新
    # ==================================================================== #

    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),  # mask token种类数 = mask策略数
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    # 目标编码器是编码器的深度拷贝
    target_encoder = copy.deepcopy(encoder)

    # ==================================================================== #
    # 第四部分：创建数据增强和数据加载器
    # ==================================================================== #

    # 创建mask生成器
    if mask_type == 'multiblock3d':
        logger.info('Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    else:
        logger.info('Initializing random tube mask')
        mask_collator = TubeMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)

    # 创建视频数据增强pipeline
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size)

    # 创建数据加载器
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         data=dataset_type,
         root_path=dataset_paths,
         batch_size=batch_size,
         training=True,
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         datasets_weights=datasets_weights,
         collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=folder if log_resource_util_data else None)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataest length: {ipe}/{_dlen}')

    # ==================================================================== #
    # 第五部分：初始化优化器、学习率调度器
    # ==================================================================== #

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps)

    # 用DistributedDataParallel包装模型（支持多GPU训练）
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    # 目标编码器不需要梯度（通过EMA更新）
    for p in target_encoder.parameters():
        p.requires_grad = False

    # ==================================================================== #
    # 第六部分：初始化EMA动量调度器
    # ==================================================================== #
    # EMA (Exponential Moving Average)：目标编码器的参数通过EMA从编码器更新
    # target_param = m * target_param + (1-m) * encoder_param
    # m 是动量，从 ema[0] 线性增加到 ema[1]（如从0.998到1.0）
    # 大的动量意味着目标编码器更新很慢，提供稳定的训练目标
    # ==================================================================== #
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    # ==================================================================== #
    # 第七部分：加载checkpoint（如果存在）
    # ==================================================================== #
    start_epoch = 0
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        # 将调度器推进到当前epoch
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, path):
        """保存训练checkpoint（只有rank 0进程保存）"""
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    # ==================================================================== #
    # 第八部分：主训练循环
    # ==================================================================== #
    logger.info('Initializing loader...')
    loader = iter(unsupervised_loader)

    # 可选的跳过batch（用于调试或恢复训练）
    if skip_batches > 0:
        logger.info(f'Skip {skip_batches} batches')
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f'Skip {itr}/{skip_batches} batches')
            try:
                udata = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                udata = next(loader)

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # 更新分布式采样器的epoch（保证每个epoch的数据shuffle不同）
        unsupervised_sampler.set_epoch(epoch)

        # 初始化各指标的统计器
        loss_meter = AverageMeter()         # 总loss
        input_var_meter = AverageMeter()    # 输入方差
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()    # JEPA预测loss
        reg_loss_meter = AverageMeter()     # 正则化loss
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            # 加载一个batch的数据和mask
            try:
                udata, masks_enc, masks_pred = next(loader)
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'

            def load_clips():
                """
                将视频数据加载到GPU

                V-JEPA中一个样本可能被多种mask策略处理：
                - masks_enc: 列表，每个元素是一种策略的编码器mask
                - masks_pred: 列表，每个元素是一种策略的预测器mask
                - 同一个视频会被每种mask策略独立处理
                - num_clips: 每个视频采样多个clip，增加数据利用率
                """
                # 将所有clip拼接成一个大的batch
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

                # 将mask也移到GPU，并对每个clip复制相同的mask
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    # 同样的mask复制给每个clip
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred)
            clips, masks_enc, masks_pred = load_clips()

            # 记录每种mask策略保留的patch数量
            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            def train_step():
                """
                单步训练的核心逻辑 —— 这就是V-JEPA的一次前向+反向传播

                每次迭代包含以下子步骤：
                1. 更新学习率和权重衰减
                2. 用目标编码器计算目标特征（无梯度）
                3. 用编码器+预测器计算预测特征
                4. 计算loss = |预测 - 目标|^p
                5. 反向传播，更新编码器和预测器
                6. 用EMA更新目标编码器
                """
                _new_lr = scheduler.step()      # 更新学习率
                _new_wd = wd_scheduler.step()   # 更新权重衰减
                # --

                def forward_target(c):
                    """
                    【目标分支】用目标编码器处理完整视频，提取目标特征

                    目标编码器的特点：
                    - 看到完整的视频（没有mask）
                    - 参数通过EMA更新（不参与梯度下降）
                    - 提供稳定的训练目标，防止模型坍缩

                    返回：目标特征列表（每种mask策略一个）
                    """
                    with torch.no_grad():  # 目标编码器不计算梯度
                        h = target_encoder(c)  # 完整视频的特征 [B, N, D]
                        h = F.layer_norm(h, (h.size(-1),))  # 在特征维做归一化
                        # 只保留目标区域的token（被遮罩的区域）
                        h = apply_masks(h, masks_pred, concat=False)
                        return h

                def forward_context(c, h):
                    """
                    【上下文分支】用编码器处理上下文区域，预测器预测目标特征

                    流程：
                    1. 编码器只看context区域 → 生成上下文特征 z
                    2. 预测器结合上下文特征z和目标特征h → 预测目标区域特征
                    """
                    z = encoder(c, masks_enc)    # 编码器：只看上下文
                    z = predictor(z, h, masks_enc, masks_pred)  # 预测器：预测目标
                    return z

                def loss_fn(z, h):
                    """
                    计算 V-JEPA 的核心损失函数

                    loss = mean(|z - h|^p) / p
                    z: 预测的特征
                    h: 目标特征（来自目标编码器）

                    p = loss_exp:
                      p=1: L1 loss（更鲁棒，V-JEPA默认使用）
                      p=2: L2 loss（MSE，对异常值更敏感）
                    """
                    loss = 0.
                    for zi, hi in zip(z, h):  # 对每种mask策略分别计算loss
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(masks_pred)  # 平均所有mask策略的loss
                    return loss

                def reg_fn(z):
                    """
                    正则化：防止预测器的输出方差过小（防止坍缩）

                    鼓励预测器在不同patch上产生多样化的预测。
                    如果所有预测都趋同，方差 → 0，正则化项 → 大。
                    """
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # ---- 步骤1: 前向传播 ----
                loss_jepa, loss_reg = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)     # 目标特征
                    z = forward_context(clips, h)  # 预测特征
                    loss_jepa = loss_fn(z, h)      # JEPA预测损失
                    pstd_z = reg_fn(z)             # 预测方差（用于正则化）
                    loss_reg += torch.mean(F.relu(1.-pstd_z))  # 鼓励方差≥1
                loss = loss_jepa + reg_coeff * loss_reg  # 总损失

                # ---- 步骤2: 反向传播 ----
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                # 优化器步进
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # 记录梯度统计
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optimizer.zero_grad()  # 清空梯度
                optim_stats = adamw_logger(optimizer)  # AdamW状态统计

                # ---- 步骤3: EMA更新目标编码器 ----
                # target_param = m * target_param + (1-m) * encoder_param
                m = next(momentum_scheduler)  # 当前EMA动量
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )

            # 执行训练步骤并计时
            (loss, loss_jepa, loss_reg, _new_lr, _new_wd,
             grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.

            # 更新各指标
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # 定期打印日志
            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss, loss_jepa, loss_reg,
                    grad_stats.global_norm, grad_stats_pred.global_norm,
                    gpu_etime_ms, iter_elapsed_time_ms)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | p%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch + 1, itr, loss_meter.avg, jepa_loss_meter.avg,
                           reg_loss_meter.avg, input_var_meter.avg, input_var_min_meter.avg,
                           '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                           _new_wd, _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.0**2,
                           gpu_time_meter.avg, wall_time_meter.avg))

                    if optim_stats is not None:
                        logger.info(
                            '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                            % (epoch + 1, itr, optim_stats.get('exp_avg').avg,
                               optim_stats.get('exp_avg').min, optim_stats.get('exp_avg').max,
                               optim_stats.get('exp_avg_sq').avg,
                               optim_stats.get('exp_avg_sq').min, optim_stats.get('exp_avg_sq').max))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr, grad_stats.first_layer, grad_stats.last_layer,
                               grad_stats.min, grad_stats.max, grad_stats.global_norm))

                    if grad_stats_pred is not None:
                        logger.info(
                            '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr, grad_stats_pred.first_layer, grad_stats_pred.last_layer,
                               grad_stats_pred.min, grad_stats_pred.max, grad_stats_pred.global_norm))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'  # 如果loss是NaN则停止训练

        # ---- 每个epoch结束后保存checkpoint ----
        logger.info('avg. loss %.3f' % loss_meter.avg)
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f'{tag}-e{epoch}.pth.tar'
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
