# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 图像分类评估（冻结backbone）
# ============================================================================
# 这个文件实现了 V-JEPA 的标准评估协议 —— "冻结评估(frozen evaluation)"。
#
# 核心思想：
# 1. 加载预训练的V-JEPA编码器（如ViT-Huge在视频上预训练后）
# 2. 冻结编码器的所有参数（不更新）
# 3. 在编码器之上添加一个轻量级的注意力分类头(AttentiveClassifier)
# 4. 只训练分类头的参数
# 5. 用分类准确率来衡量编码器学到的表征质量
#
# 这种评估方式的意义：
# - 如果只用一个小分类头就能取得好结果，说明编码器学到了很好的表征
# - 避免了"微调整个模型"带来的混淆（不清楚是预训练好还是微调过程好）
# - 计算开销小，可以快速评估

import os

# -- 分布式训练：每个进程只看到自己的GPU
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint
import numpy as np

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel

from timm.data import create_transform as timm_make_transforms

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed, AllReduce
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.logging import AverageMeter, CSVLogger

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
    图像分类冻结评估的主函数

    流程:
    1. 从配置文件中读取参数
    2. 加载预训练的编码器并冻结
    3. 创建注意力分类头
    4. 创建数据加载器（训练集+验证集）
    5. 训练分类头（编码器保持冻结）
    6. 记录分类准确率
    """

    # ================================================================ #
    # 第一部分：解析配置参数
    # ================================================================ #

    # -- 预训练模型参数
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    # checkpoint_key: 使用encoder还是target_encoder的参数
    # 通常用target_encoder（EMA后的参数更稳定）
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
    # 视频模型特有参数
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- 数据集参数
    args_data = args_eval.get('data')
    dataset_name = args_data.get('dataset_name')     # 如 'ImageNet'
    num_classes = args_data.get('num_classes')         # 如 1000
    root_path = args_data.get('root_path', None)
    image_folder = args_data.get('image_folder', None)
    resolution = args_data.get('resolution', 224)

    # -- 优化参数
    args_opt = args_eval.get('optimization')
    batch_size = args_opt.get('batch_size')
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')

    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ================================================================ #
    # 第二部分：初始化环境
    # ================================================================ #

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

    # 日志和checkpoint路径
    folder = os.path.join(pretrain_folder, 'image_classification_frozen/')
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

    # ================================================================ #
    # 第三部分：加载预训练编码器并冻结
    # ================================================================ #

    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)

    # 冻结编码器参数（不计算梯度，不更新）
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 创建分类头（只有这部分参数会被训练）
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,                 # 浅层分类头
        num_classes=num_classes
    ).to(device)

    # ================================================================ #
    # 第四部分：创建数据加载器
    # ================================================================ #

    train_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True)
    val_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False)
    ipe = len(train_loader)

    # ================================================================ #
    # 第五部分：初始化优化器
    # ================================================================ #

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

    # 可选的checkpoint恢复
    start_epoch = 0
    if resume_checkpoint:
        classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device, r_path=latest_path,
            classifier=classifier, opt=optimizer, scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch, 'batch_size': batch_size,
            'world_size': world_size, 'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # ================================================================ #
    # 第六部分：训练循环
    # ================================================================ #

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # 训练一个epoch
        train_acc = run_one_epoch(
            device=device, training=True,
            encoder=encoder, classifier=classifier,
            scaler=scaler, optimizer=optimizer,
            scheduler=scheduler, wd_scheduler=wd_scheduler,
            data_loader=train_loader, use_bfloat16=use_bfloat16)

        # 验证一个epoch
        val_acc = run_one_epoch(
            device=device, training=False,
            encoder=encoder, classifier=classifier,
            scaler=scaler, optimizer=optimizer,
            scheduler=scheduler, wd_scheduler=wd_scheduler,
            data_loader=val_loader, use_bfloat16=use_bfloat16)

        logger.info('[%5d] train: %.3f%% test: %.3f%%' % (epoch + 1, train_acc, val_acc))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc)
        save_checkpoint(epoch + 1)


def run_one_epoch(
    device, training, encoder, classifier, scaler,
    optimizer, scheduler, wd_scheduler, data_loader, use_bfloat16,
):
    """
    运行一个epoch的训练或验证

    关键点：
    - 编码器始终在torch.no_grad()下运行（冻结，不计算梯度）
    - 分类头在训练时计算梯度，验证时不计算
    - 使用交叉熵损失(CrossEntropyLoss)做多分类
    """

    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()

    for itr, data in enumerate(data_loader):
        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            imgs, labels = data[0].to(device), data[1].to(device)

            # 编码器前向（无梯度）
            with torch.no_grad():
                outputs = encoder(imgs)
                # 训练时：先计算编码器特征，再输入分类头（节省显存）
                # 验证时：直接用整个pipeline
                if not training:
                    outputs = classifier(outputs)

            if training:
                outputs = classifier(outputs)

        # 计算损失和准确率
        loss = criterion(outputs, labels)
        top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / len(imgs)
        top1_acc = float(AllReduce.apply(top1_acc))  # 跨GPU平均准确率
        top1_meter.update(top1_acc)

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
    """加载分类头checkpoint"""
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
        del checkpoint
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0
    return classifier, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key='target_encoder'):
    """
    加载预训练的编码器参数

    处理不同来源checkpoint的兼容性：
    - checkpoint可能有 encoder/target_encoder 两个key
    - 参数名可能有 module./backbone. 前缀（DDP和多mask包装器）
    """
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    # 去除DDP和多mask包装器的前缀
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}

    # 处理形状不匹配的参数
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v

    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    dataset_name, root_path, image_folder, batch_size,
    world_size, rank, resolution=224, training=False, subset_file=None
):
    """
    创建图像数据加载器

    训练时：使用timm的自动增强（RandAugment + RandomErasing）
    验证时：使用简单的Resize + CenterCrop
    """
    normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if training:
        logger.info('implementing auto-agument strategy')
        transform = timm_make_transforms(
            input_size=resolution,
            is_training=training,
            auto_augment='original',
            interpolation='bicubic',
            re_prob=0.25,          # 25%概率随机擦除
            re_mode='pixel',
            re_count=1,
            mean=normalization[0],
            std=normalization[1])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=int(resolution * 256/224)),
            transforms.CenterCrop(size=resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1])])

    data_loader, _ = init_data(
        data=dataset_name, transform=transform,
        batch_size=batch_size, world_size=world_size, rank=rank,
        root_path=root_path, image_folder=image_folder,
        training=training, copy_data=False, drop_last=False,
        subset_file=subset_file)
    return data_loader


def init_model(
    device, pretrained, model_name, patch_size=16, crop_size=224,
    frames_per_clip=16, tubelet_size=2, use_sdpa=False,
    use_SiLU=False, tight_SiLU=True, uniform_power=False,
    checkpoint_key='target_encoder'
):
    """
    初始化预训练的编码器

    对于视频预训练模型(frames_per_clip > 1)用于图像分类：
    使用forward_prehook将2D图像自动扩展为3D视频格式。
    例如：[B, 3, 224, 224] → [B, 3, 16, 224, 224]（重复16帧）
    这样视频预训练的模型就能处理单张图像了。
    """
    encoder = vit.__dict__[model_name](
        img_size=crop_size, patch_size=patch_size,
        num_frames=frames_per_clip, tubelet_size=tubelet_size,
        uniform_power=uniform_power, use_sdpa=use_sdpa,
        use_SiLU=use_SiLU, tight_SiLU=tight_SiLU,
    )

    if frames_per_clip > 1:
        # 前向钩子：自动将图像扩展为视频
        def forward_prehook(module, input):
            input = input[0]  # [B, C, H, W]
            input = input.unsqueeze(2).repeat(1, 1, frames_per_clip, 1, 1)  # → [B, C, T, H, W]
            return (input)
        encoder.register_forward_pre_hook(forward_prehook)

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def init_opt(
    classifier, iterations_per_epoch, start_lr, ref_lr,
    warmup, num_epochs, wd=1e-6, final_wd=1e-6,
    final_lr=0.0, use_bfloat16=False
):
    """
    初始化优化器（仅训练分类头）

    参数分组：
    - 权重参数：使用权重衰减
    - bias和1D参数：不使用权重衰减
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
        optimizer, warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=wd, final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
