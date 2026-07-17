# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
V-JEPA Physion++ 注意力探针评估脚本

在 Physion++ 数据集上，使用冻结的 V-JEPA 预训练编码器 + 注意力探针（AttentiveClassifier）
进行 OCP（Object Contact Prediction，物体接触预测）任务的评估。

【实验设计】
  1. 加载预训练的 V-JEPA 编码器（冻结所有参数）
  2. 在 Physion++ readout_data 上训练 AttentiveClassifier 探针
  3. 在 Physion++ test_data 上评估探针性能
  4. 可选：与 human_data 对比

【注意力探针 (Attentive Probe)】
  使用 AttentiveClassifier（= AttentivePooler + Linear）作为探针：
  - AttentivePooler: 通过交叉注意力机制，从 encoder 输出的 patch 特征中
    自适应地汇聚与物理接触预测相关的信息
  - Linear: 将汇聚后的特征映射到二分类输出（接触/不接触）

  与简单平均池化的区别：
    平均池化对所有 patch 一视同仁，但 OCP 任务中只有特定区域
    （如物体接触点附近）的信息是关键的。AttentivePooler 通过
    可学习的查询向量自动发现这些关键区域。

【Physion++ OCP 任务】
  - 输入：一段视频，展示两个物体在物理仿真场景中运动
  - 输出：二分类 —— 物体是否会接触？（YES=1 / NO=0）
  - 4 种物理属性：mass（质量）、friction（摩擦力）、elasticity（弹性）、deformability（可变形性）
  - 每种属性独立评估

【评估模式】
  1. joint（联合模式）：所有 4 种属性混合训练一个探针
  2. per_property（分属性模式）：每种属性独立训练一个探针（默认）
  3. multi_head（多头模式）：共享 encoder，4 个独立的分类头

【主要流程】
  1. main(): 解析配置 → 构建模型 → 创建数据加载器 → 训练探针 → 评估
  2. 每种物理属性独立进行训练和评估
  3. 最终汇总所有属性的结果
"""

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.distributed import init_distributed, AllReduce
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.logging import AverageMeter, CSVLogger

from evals.physion_attentive_probe.utils import make_physion_transforms
from src.datasets.physion_dataset import (
    make_physion_dataset,
    PHYSION_PROPERTIES,
    PROPERTY_TO_INDEX,
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):
    """
    Physion++ 注意力探针评估主函数。

    对每种物理属性独立训练和评估一个 AttentiveClassifier 探针。

    参数:
        args_eval (dict): YAML 配置参数字典
        resume_preempt (bool): 是否从检查点恢复训练
    """

    # ========================================================================
    # 参数提取
    # ========================================================================

    # ---- 预训练模型参数 ----
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # ---- 数据参数 ----
    args_data = args_eval.get('data')
    train_data_path = [args_data.get('dataset_train')]
    val_data_path = [args_data.get('dataset_val')]
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_data.get('frame_step', 4)
    eval_duration = args_data.get('clip_duration', None)

    # ---- 评估模式 ----
    eval_mode = args_eval.get('eval_mode', 'per_property')
    # eval_mode: 评估模式
    #   'per_property' (默认): 每种物理属性独立训练和评估一个探针
    #   'joint': 所有属性混合，训练一个统一的探针
    properties_to_eval = args_eval.get('properties', None)
    # properties_to_eval: 要评估的属性列表
    #   None → 评估全部 4 种属性
    #   ['mass', 'friction'] → 只评估指定属性

    # ---- 优化参数 ----
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    reprob = args_opt.get('reprob', 0.0)
    auto_augment = args_opt.get('auto_augment', False)
    random_horizontal_flip = args_opt.get('random_horizontal_flip', False)

    # ---- 探针参数 ----
    probe_depth = args_eval.get('probe_depth', 1)
    probe_complete_block = args_eval.get('probe_complete_block', True)

    # ---- 实验标识 ----
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ========================================================================
    # 分布式初始化
    # ========================================================================

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # ---- 日志和检查点路径 ----
    folder = os.path.join(pretrain_folder, 'physion_attentive_probe/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # ========================================================================
    # 模型初始化（只创建一次编码器）
    # ========================================================================

    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,
    ).float()  # 强制 float32，避免与 autocast 类型冲突

    # 冻结编码器
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ========================================================================
    # 确定要评估的属性列表
    # ========================================================================
    if properties_to_eval is None:
        properties_to_eval = PHYSION_PROPERTIES
    logger.info(f'Evaluating properties: {properties_to_eval}')
    logger.info(f'Evaluation mode: {eval_mode}')

    # ========================================================================
    # 评估
    # ========================================================================

    # Physion++ 是二分类任务（OCP: YES/NO）
    num_classes = 2

    if eval_mode == 'joint':
        # 联合模式：所有属性混合训练一个探针
        results = _evaluate_single_property(
            device=device,
            encoder=encoder,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            resolution=resolution,
            eval_frames_per_clip=eval_frames_per_clip,
            eval_frame_step=eval_frame_step,
            eval_duration=eval_duration,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            num_epochs=num_epochs,
            wd=wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            warmup=warmup,
            use_bfloat16=use_bfloat16,
            reprob=reprob,
            auto_augment=auto_augment,
            random_horizontal_flip=random_horizontal_flip,
            num_classes=num_classes,
            probe_depth=probe_depth,
            probe_complete_block=probe_complete_block,
            folder=folder,
            tag=tag,
            resume_checkpoint=resume_checkpoint,
            properties_filter=properties_to_eval,
            property_name='all',
        )
        if rank == 0:
            _log_results(folder, tag, {'all': results})

    elif eval_mode == 'per_property':
        # 分属性模式：每种属性独立训练和评估
        all_results = {}
        for prop in properties_to_eval:
            logger.info(f'\n{"="*60}')
            logger.info(f'Evaluating property: {prop}')
            logger.info(f'{"="*60}')

            results = _evaluate_single_property(
                device=device,
                encoder=encoder,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                resolution=resolution,
                eval_frames_per_clip=eval_frames_per_clip,
                eval_frame_step=eval_frame_step,
                eval_duration=eval_duration,
                batch_size=batch_size,
                world_size=world_size,
                rank=rank,
                num_epochs=num_epochs,
                wd=wd,
                start_lr=start_lr,
                ref_lr=lr,
                final_lr=final_lr,
                warmup=warmup,
                use_bfloat16=use_bfloat16,
                reprob=reprob,
                auto_augment=auto_augment,
                random_horizontal_flip=random_horizontal_flip,
                num_classes=num_classes,
                probe_depth=probe_depth,
                probe_complete_block=probe_complete_block,
                folder=folder,
                tag=tag,
                resume_checkpoint=resume_checkpoint,
                properties_filter=[prop],
                property_name=prop,
            )
            all_results[prop] = results

        if rank == 0:
            _log_results(folder, tag, all_results)

    elif eval_mode == 'multi_head':
        # 多头模式：共享 encoder，每种属性一个独立的分类头
        # TODO: 实现多头探针（4 个 AttentiveClassifier 共享 encoder）
        raise NotImplementedError(
            'multi_head mode not yet implemented. '
            'Use per_property or joint mode.'
        )

    else:
        raise ValueError(f'Unknown eval_mode: {eval_mode}')


def _evaluate_single_property(
    device,
    encoder,
    train_data_path,
    val_data_path,
    resolution,
    eval_frames_per_clip,
    eval_frame_step,
    eval_duration,
    batch_size,
    world_size,
    rank,
    num_epochs,
    wd,
    start_lr,
    ref_lr,
    final_lr,
    warmup,
    use_bfloat16,
    reprob,
    auto_augment,
    random_horizontal_flip,
    num_classes,
    probe_depth,
    probe_complete_block,
    folder,
    tag,
    resume_checkpoint,
    properties_filter,
    property_name,
):
    """
    对特定物理属性（或属性集合）训练和评估一个注意力探针。

    这是评估的核心函数，对每种属性独立调用。

    参数:
        properties_filter: 要加载的属性列表（用于数据过滤）
        property_name: 属性名称，用于日志和保存检查点
        其余参数含义见 main() 函数

    返回:
        dict: 包含训练和测试准确率的结果字典
    """

    # ---- 创建探针分类器 ----
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=probe_depth,
        num_classes=num_classes,
        complete_block=probe_complete_block,
    ).to(device)

    # ---- 创建数据加载器 ----
    train_loader, val_loader = _make_physion_dataloaders(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        properties_filter=properties_filter,
        reprob=reprob,
        auto_augment=auto_augment,
        random_horizontal_flip=random_horizontal_flip,
    )
    ipe = len(train_loader)
    logger.info(f'Property [{property_name}]: {ipe} iterations per epoch')

    # ---- 优化器 ----
    optimizer, scaler, scheduler, wd_scheduler = _init_physion_opt(
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )
    classifier = DistributedDataParallel(classifier, static_graph=True)

    # ---- 检查点路径 ----
    latest_path = os.path.join(
        folder, f'{tag}_property_{property_name}_latest.pth.tar'
    )
    log_file = os.path.join(
        folder, f'{tag}_property_{property_name}_r{rank}.csv'
    )

    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ('%d', 'epoch'),
            ('%.5f', 'train_loss'),
            ('%.5f', 'train_acc'),
            ('%.5f', 'val_loss'),
            ('%.5f', 'val_acc'),
        )

    # ---- 恢复训练 ----
    start_epoch = 0
    if resume_checkpoint:
        classifier, optimizer, scaler, start_epoch = _load_physion_checkpoint(
            device=device,
            r_path=latest_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': ref_lr,
        }
        if rank == 0:
            torch.save(save_dict, path)

    # ========================================================================
    # 训练循环
    # ========================================================================

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        logger.info(f'[{property_name}] Epoch {epoch + 1}/{num_epochs}')

        # 训练
        train_loss, train_acc = run_physion_epoch(
            device=device,
            training=True,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
        )

        # 验证
        val_loss, val_acc = run_physion_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
        )

        logger.info(
            f'[{property_name}] Epoch {epoch + 1:3d}: '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.2f}% '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.2f}%'
        )

        if rank == 0:
            csv_logger.log(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # 保存最优检查点
            best_path = os.path.join(
                folder, f'{tag}_property_{property_name}_best.pth.tar'
            )
            save_checkpoint(epoch + 1, best_path)

        # 无论是否最优，始终保存最新检查点（用于断点续训）
        save_checkpoint(epoch + 1, latest_path)

    logger.info(
        f'[{property_name}] Best val_acc: {best_val_acc:.2f}% '
        f'at epoch {best_epoch}'
    )

    return {
        'property': property_name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
    }


def run_physion_epoch(
    device,
    training,
    encoder,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
):
    """
    执行一个 epoch 的 Physion++ 探针训练或验证。

    与 video_classification_frozen/run_one_epoch 的核心区别：
    - Physion++ 数据格式更简单：单 clip、单视角
    - 二分类任务而非多分类
    - 无需多视角聚合

    参数:
        device: 计算设备
        training (bool): 训练/验证模式
        encoder: 冻结的 V-JEPA 编码器
        classifier: 可训练的 AttentiveClassifier 探针
        scaler: AMP 梯度缩放器
        optimizer: 优化器
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        data_loader: 数据加载器
        use_bfloat16 (bool): 是否使用混合精度

    返回:
        tuple: (平均损失, 平均准确率 %)
    """

    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # ---- Physion++ 数据格式 ----
            # data[0]: 视频帧 [B, C, T, H, W]（单 clip、单视角）
            # data[1]: OCP 标签 [B]（0=不接触, 1=接触）
            clips = data[0]
            if isinstance(clips, list):
                # 处理可能的嵌套格式
                if isinstance(clips[0], list):
                    clips = clips[0][0]
                else:
                    clips = clips[0]
            labels = data[1].to(device)
            batch_size = len(labels)

            # ---- 数据解包（Physion++ 数据可能嵌套在 list 中） ----
            if isinstance(clips, list):
                while isinstance(clips, list) and len(clips) > 0:
                    clips = clips[0]

            # ---- 前向传播 ----
            # V-JEPA 官方检查点权重是 float16，输入也转为 float16
            with torch.no_grad():
                model_dtype = next(encoder.parameters()).dtype
                clips = clips.to(device=device, dtype=model_dtype)
                features = encoder(clips)

            # Classifier/Probe: [B, N_patches, D] → [B, 2]
            logits = classifier(features)

        # ---- 损失 ----
        loss = criterion(logits, labels)

        # ---- 准确率 ----
        with torch.no_grad():
            acc = 100. * logits.max(dim=1).indices.eq(labels).sum() / batch_size
            acc = float(AllReduce.apply(acc))
            loss_meter.update(float(AllReduce.apply(loss)))
            acc_meter.update(acc)

        # ---- 反向传播 ----
        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 20 == 0:
            logger.info(
                '[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                % (itr, acc_meter.avg, loss,
                   torch.cuda.max_memory_allocated() / 1024.**2)
            )

    return loss_meter.avg, acc_meter.avg


def _make_physion_dataloaders(
    train_data_path,
    val_data_path,
    resolution,
    frames_per_clip,
    frame_step,
    eval_duration,
    batch_size,
    world_size,
    rank,
    properties_filter,
    reprob,
    auto_augment,
    random_horizontal_flip,
):
    """
    创建 Physion++ 训练和验证数据加载器。

    返回:
        tuple: (train_loader, val_loader)
    """

    # 训练数据变换
    train_transform = make_physion_transforms(
        training=True,
        crop_size=resolution,
        random_horizontal_flip=random_horizontal_flip,
        reprob=reprob,
        auto_augment=auto_augment,
        num_views_per_clip=1,
    )

    # 验证数据变换
    val_transform = make_physion_transforms(
        training=False,
        crop_size=resolution,
        random_horizontal_flip=False,
        num_views_per_clip=1,
    )

    # 训练数据加载器
    train_loader, _ = make_physion_dataset(
        data_paths=train_data_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        duration=eval_duration,
        transform=train_transform,
        shared_transform=None,
        rank=rank,
        world_size=world_size,
        num_workers=8,
        pin_mem=True,
        drop_last=False,
        properties=properties_filter,
        return_property_label=False,
    )

    # 验证数据加载器
    val_loader, _ = make_physion_dataset(
        data_paths=val_data_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=1,
        random_clip_sampling=False,
        allow_clip_overlap=False,
        duration=eval_duration,
        transform=val_transform,
        shared_transform=None,
        rank=rank,
        world_size=world_size,
        num_workers=8,
        pin_mem=True,
        drop_last=False,
        properties=properties_filter,
        return_property_label=False,
    )

    return train_loader, val_loader


def _init_physion_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
):
    """
    初始化探针训练的优化器、调度器和 AMP 缩放器。
    """

    param_groups = [
        {
            'params': (
                p for n, p in classifier.named_parameters()
                if ('bias' not in n) and (len(p.shape) != 1)
            )
        },
        {
            'params': (
                p for n, p in classifier.named_parameters()
                if ('bias' in n) or (len(p.shape) == 1)
            ),
            'WD_exclude': True,
            'weight_decay': 0,
        },
    ]

    logger.info('Using AdamW optimizer for Physion++ probe')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def _load_physion_checkpoint(device, r_path, classifier, opt, scaler):
    """从磁盘加载探针训练检查点。"""
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        pretrained_dict = checkpoint['classifier']
        # 去掉 DDP 包装的 'module.' 前缀
        pretrained_dict = {
            k.replace('module.', ''): v for k, v in pretrained_dict.items()
        }
        msg = classifier.load_state_dict(pretrained_dict, strict=False)
        logger.info(
            f'Loaded pretrained classifier from epoch {epoch} with msg: {msg}'
        )

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'Loaded optimizers from epoch {epoch}')
        logger.info(f'Read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        epoch = 0

    return classifier, opt, scaler, epoch


def _log_results(folder, tag, all_results):
    """
    记录所有属性的评估结果。

    输出汇总到控制台和 results.txt 文件。
    """

    results_path = os.path.join(folder, f'{tag}_results.txt')
    lines = []
    lines.append('=' * 60)
    lines.append('Physion++ Attentive Probe Evaluation Results')
    lines.append('=' * 60)

    total_acc = 0.0
    for prop, result in all_results.items():
        acc = result['best_val_acc']
        total_acc += acc
        lines.append(
            f'  {prop:20s}: {acc:.2f}% (best epoch: {result["best_epoch"]})'
        )

    avg_acc = total_acc / len(all_results)
    lines.append('-' * 60)
    lines.append(f'  {"Average":20s}: {avg_acc:.2f}%')
    lines.append('=' * 60)

    for line in lines:
        logger.info(line)

    with open(results_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f'Results saved to: {results_path}')


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder',
):
    """
    创建并加载预训练的 V-JEPA 视频 Vision Transformer 编码器。
    """
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )
    encoder.to(device)
    encoder = load_pretrained(
        encoder=encoder,
        pretrained=pretrained,
        checkpoint_key=checkpoint_key,
    )
    return encoder


def load_pretrained(encoder, pretrained, checkpoint_key='target_encoder'):
    """
    加载 V-JEPA 预训练权重到编码器。
    """
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {
        k.replace('module.', ''): v for k, v in pretrained_dict.items()
    }
    pretrained_dict = {
        k.replace('backbone.', ''): v for k, v in pretrained_dict.items()
    }
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" shape mismatch: '
                f'{pretrained_dict[k].shape} vs {v.shape}'
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'Loaded pretrained encoder with msg: {msg}')
    logger.info(
        f'Loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n'
        f'Path: {pretrained}'
    )
    del checkpoint
    return encoder
