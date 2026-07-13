# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 视频分类评估脚本（冻结特征主干网络）
# ============================================================================
#
# 本文件实现了基于冻结的 V-JEPA 预训练编码器进行视频分类评估的完整流程。
#
# 【核心概念 —— 视频 vs 图像评估的区别】
#   视频多了一维时间信息。V-JEPA 的视频编码器同时处理空间和时间维度的信息。
#   评估时的关键问题是：如何处理变长的视频？
#      - 采样多个"片段"（segments/clips）来覆盖整个视频
#      - 分段独立编码，然后聚合特征
#
# 【两种聚合策略】
#   1. FrameAggregation: 逐帧独立编码（将每帧当作独立图像处理），然后拼接所有帧的特征
#      适用于单帧预训练的模型（pretrain_frames_per_clip == 1）
#
#   2. ClipAggregation: 逐 clip 独立编码（将多帧 clip 一起编码，利用 tubelet 处理时间变化）
#      适用于视频预训练的模型（pretrain_frames_per_clip > 1）
#      可选 attend_across_segments，让分类器跨片段进行注意力交互
#
# 【多头评估 (Multi-View Evaluation)】
#   空间多头 (Spatial Views)：从同一帧中裁剪多个视角区域，取平均预测
#   时间多头 (Temporal Views/Segments)：采样视频中的多个时间片段，取平均预测
#   最终预测 = 所有空间和时间视角预测的平均值
#
# 【主要流程】
#   1. main(): 解析配置 → 选择聚合策略 → 初始化模型 → 多视角数据加载 → 训练循环
#   2. run_one_epoch(): 处理多视角数据的前向传播和聚合
#   3. load_pretrained(): 加载预训练视频编码器权重
#   4. init_model(): 创建视频 Vision Transformer 编码器
#   5. init_opt(): 初始化优化器（与图像评估相同）
#   6. make_dataloader(): 创建视频数据加载器（支持多视角采样）
# ============================================================================

import os

# ---- 分布式训练：确保每个进程只使用一张 GPU ----
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F               # 包含 softmax 等函数

from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.datasets.data_manager import init_data
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

# 视频评估专用的工具函数
from evals.video_classification_frozen.utils import (
    make_transforms,       # 创建视频数据增强变换
    ClipAggregation,       # 逐 clip 独立编码并聚合
    FrameAggregation       # 逐帧独立编码并聚合
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
    视频分类评估的主函数。

    整体流程与图像评估类似，但有以下关键区别：
    - 数据是三维的（C×T×H×W），包含时间维度
    - 支持时间片段采样（temporal segments）和空间多视角（spatial views）
    - 使用 FrameAggregation 或 ClipAggregation 处理变长视频
    - 支持跨片段注意力（attend_across_segments）

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
    # 视频模型参数
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    # tubelet_size: 时间维度上的扩展，将连续的 tubelet_size 帧合并为一个 token
    # 例如，tubelet_size=2 表示每 2 帧被组合在一起处理
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)
    # pretrain_frames_per_clip: 预训练时每个 clip 的帧数
    #   1 表示逐帧预训练（类似图像模型）
    #   16 表示以 16 帧 clip 为单位预训练

    # ---- 数据参数 ----
    args_data = args_eval.get('data')
    train_data_path = [args_data.get('dataset_train')]    # 训练集路径
    val_data_path = [args_data.get('dataset_val')]         # 验证集路径
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')              # 动作类别数
    # 评估时的采样参数
    eval_num_segments = args_data.get('num_segments', 1)
    # num_segments: 从整个视频中均匀采样多少个时间片段
    # 例如 10，表示从 10 秒的视频中均匀采样 10 个片段
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    # frames_per_clip: 每个片段包含多少帧
    eval_frame_step = args_pretrain.get('frame_step', 4)
    # frame_step: 采样帧的步长（每隔多少帧取一帧）
    eval_duration = args_pretrain.get('clip_duration', None)
    # clip_duration: 每个片段的持续时间（秒），None 表示使用帧数
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)
    # num_views_per_segment: 每个时间片段裁多少个空间视角
    # =1 是标准的中心裁剪，>1 会进行多视角评估

    # ---- 优化参数 ----
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    # attend_across_segments: 是否让分类器跨时间片段进行注意力交互
    # False: 每个片段独立分类，然后平均 softmax 概率（标准做法）
    # True:  所有片段的特征拼接在一起，通过一次注意力分类（允许跨片段信息交互）
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')

    # ---- 实验标识 ----
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ========================================================================

    # spawn 模式（CUDA 安全）
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # 设备选择
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # 分布式初始化
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # 日志和检查点路径
    folder = os.path.join(pretrain_folder, 'video_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    if rank == 0:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%.5f', 'loss'),
                               ('%.5f', 'acc'))

    # ========================================================================
    # 模型初始化
    # ========================================================================

    # 创建预训练编码器
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
        use_sdpa=use_sdpa)

    # ---- 选择聚合策略 ----
    # 根据预训练时的帧处理方式来选择：
    #   pretrain_frames_per_clip == 1 → 模型学的是逐帧特征 → FrameAggregation
    #   pretrain_frames_per_clip > 1  → 模型学的是 clip 特征 → ClipAggregation
    if pretrain_frames_per_clip == 1:
        # 逐帧编码：将视频的每一帧独立送入编码器，然后拼接所有帧的特征
        encoder = FrameAggregation(encoder).to(device)
    else:
        # 逐 clip 编码：将视频分成多个 clip，每个 clip 独立编码
        # attend_across_segments 决定是否跨片段交互
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)

    # 冻结编码器参数
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 分类器
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    # ========================================================================
    # 数据加载器
    # ========================================================================

    # 训练数据加载器
    # 注意：训练时通常只需要 1 个时间片段（随机采样一个位置）
    # attend_across_segments=True 时例外，需要多个片段
    train_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        num_segments=eval_num_segments if attend_across_segments else 1,
        num_views_per_segment=1,       # 训练时只用 1 个空间视角
        allow_segment_overlap=True,    # 允许片段之间有时间重叠
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True)

    # 验证数据加载器
    # 验证时使用全部时间片段和空间视角，通过聚合提高准确性
    val_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_segments=eval_num_segments,          # 多个时间片段
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,  # 多个空间视角
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False)
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # ========================================================================
    # 优化器
    # ========================================================================
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)
    classifier = DistributedDataParallel(classifier, static_graph=True)

    # 恢复训练
    start_epoch = 0
    if resume_checkpoint:
        classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # ========================================================================
    # 训练循环
    # ========================================================================
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_acc = run_one_epoch(
            device=device,
            training=True,
            num_temporal_views=eval_num_segments if attend_across_segments else 1,
            attend_across_segments=attend_across_segments,
            num_spatial_views=1,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16)

        val_acc = run_one_epoch(
            device=device,
            training=False,
            num_temporal_views=eval_num_segments,
            attend_across_segments=attend_across_segments,
            num_spatial_views=eval_num_views_per_segment,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16)

        logger.info('[%5d] train: %.3f%% test: %.3f%%' % (epoch + 1, train_acc, val_acc))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc)
        save_checkpoint(epoch + 1)


def run_one_epoch(
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
    num_spatial_views,
    num_temporal_views,
    attend_across_segments,
):
    """
    执行一个 epoch 的视频评估（训练或验证）。

    与图像评估的关键区别在于数据的组织方式：
    - 视频数据以嵌套列表形式组织：[时间片段][空间视角]
    - 每个元素是一个 5D 张量 [B, C, T, H, W]
    - 需要对多视角预测进行聚合（平均 softmax 概率）

    参数:
        device: 计算设备
        training (bool): 训练/验证模式
        encoder: 冻结的编码器（带聚合策略）
        classifier: 可训练的分类器
        scaler: AMP 梯度缩放器
        optimizer: 优化器
        scheduler: 学习率调度器
        wd_scheduler: 权重衰减调度器
        data_loader: 数据加载器
        use_bfloat16: 是否使用混合精度
        num_spatial_views (int): 空间视角数
        num_temporal_views (int): 时间视角数
        attend_across_segments (bool): 是否跨片段注意力

    返回:
        float: 平均 top-1 准确率
    """

    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()

    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # ---- 数据预处理 ----
            # 视频数据以嵌套列表形式组织：
            #   data[0]: 视频帧数据，结构为 List[temporal_view][spatial_view]
            #            每个元素是 Tensor[B, C, T, H, W]
            #   data[1]: 标签，Tensor[B]
            #   data[2]: 片段索引（用于位置编码），List[temporal_view]
            #            每个元素是 Tensor[B, num_frames]
            # non_blocking=True: 异步将数据传到 GPU，允许 CPU-GPU 传输与计算重叠
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # 遍历空间视角
                for di in data[0]                                    # 遍历时间视角
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)

            # ---- 前向传播 ----
            # 编码器输出是嵌套列表：[空间视角] 或 [空间视角][时间视角]
            #   attend_across_segments=False: 输出为 List[spatial][temporal]
            #   attend_across_segments=True:  输出为 List[spatial]（已拼接所有 temporal）
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    if attend_across_segments:
                        # 每个空间视角的输出已被编码器拼接好，直接分类
                        outputs = [classifier(o) for o in outputs]
                    else:
                        # 每个 (空间, 时间) 组合独立分类
                        outputs = [[classifier(ost) for ost in os] for os in outputs]
            if training:
                if attend_across_segments:
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os] for os in outputs]

        # ---- 损失计算 ----
        # 对所有的 (空间视角, 时间视角) 组合取平均
        if attend_across_segments:
            loss = sum([criterion(o, labels) for o in outputs]) / len(outputs)
        else:
            loss = sum([sum([criterion(ost, labels) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])

        # ---- 准确率计算 ----
        # 聚合策略：先对每个预测做 softmax 得到概率分布，再平均所有视角的概率
        # 这比直接聚合 logits 更合理，因为 softmax 将 logits 归一化到同一尺度
        with torch.no_grad():
            if attend_across_segments:
                # 多个空间视角的 softmax 取平均
                outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
            else:
                # 所有 (空间, 时间) 组合的 softmax 取平均
                outputs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
            top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)

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
            logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, top1_meter.avg, loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))

    return top1_meter.avg


def load_checkpoint(device, r_path, classifier, opt, scaler):
    """
    从磁盘加载训练检查点。详见图像评估的同名函数。
    """
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        pretrained_dict = checkpoint['classifier']
        msg = classifier.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return classifier, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key='target_encoder'):
    """
    加载 V-JEPA 预训练视频编码器的权重。详见图像评估的同名函数。
    """
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None
):
    """
    创建视频数据加载器。

    视频数据加载与图像不同：
    - 需要从视频中采样帧序列（clip）
    - 可以采样多个片段覆盖整个视频
    - 训练时随机采样，验证时均匀采样

    参数:
        root_path (list[str]): 数据集路径列表
        batch_size (int): 每个 GPU 的批次大小
        world_size (int): GPU 总数
        rank (int): 当前 GPU 排名
        dataset_type (str): 数据集类型，如 'VideoDataset'
        resolution (int): 空间分辨率
        frames_per_clip (int): 每个 clip 的帧数
        frame_step (int): 采样步长（每隔几帧采一帧）
        num_segments (int): 每个视频采样的时间片段数
        eval_duration (float): 每个片段的持续时间（秒），None 表示用 frames_per_clip
        num_views_per_segment (int): 每个片段的空间视角数
        allow_segment_overlap (bool): 是否允许时间片段重叠
        training (bool): 训练/验证模式
        num_workers (int): 数据加载的并行 worker 数
        subset_file (str): 可选的数据子集文件

    返回:
        DataLoader: 视频数据加载器
    """
    # 创建视频变换（数据增强）
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,                # 视频一般不水平翻转（会破坏方向信息）
        random_resize_aspect_ratio=(0.75, 4/3),      # 随机缩放时的宽高比范围
        random_resize_scale=(0.08, 1.0),             # 随机缩放时的尺度范围
        reprob=0.25,                                  # 随机擦除概率
        auto_augment=True,                            # 启用自动数据增强
        motion_shift=False,                           # 是否启用运动偏移增强
        crop_size=resolution,
    )

    # init_data 创建视频数据加载器
    # clip_len: 每个 clip 的帧数
    # frame_sample_rate: 帧采样步长
    # num_clips: 时间片段数
    # duration: 片段持续时间
    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # 视频专属参数
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    """
    创建视频 Vision Transformer 编码器并加载预训练权重。

    与图像 ViT 的关键区别：
    - tubelet_size: 时间维度上的扩展。将连续的 tubelet_size 帧合并为一个时空 patch
      例如，patch_size=16, tubelet_size=2 会产生 16×16×2 的时空 patch
    - num_frames: 输入视频的帧数（视频评估时需要 > 1）

    参数:
        device: 计算设备
        pretrained (str): 预训练检查点路径
        model_name (str): 模型名称
        patch_size (int): 空间 patch 大小
        crop_size (int): 空间裁剪大小
        frames_per_clip (int): 每个 clip 的帧数
        tubelet_size (int): 时间 tubelet 大小（时间维度上的 patch 大小）
        use_sdpa: 高效注意力
        use_SiLU: SiLU 激活
        tight_SiLU: 精确 SiLU
        uniform_power: 均匀幂分布初始化
        checkpoint_key: 检查点键名

    返回:
        加载了预训练权重的视频 ViT 编码器
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
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
):
    """
    初始化优化器、调度器和 AMP 缩放器。详见图像评估的同名函数。
    """
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
