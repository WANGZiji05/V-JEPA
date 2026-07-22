# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
V-JEPA Physion++ Latent Rollout 评估

============================================================
实验原理
============================================================

V-JEPA 的核心机制是在 latent 空间（而非像素空间）做预测。
预训练时：给出被 mask 视频的上下文特征 → 预测被 mask 区域的特征。

本 rollout 实验直接验证这个机制：
  - 以 Physion++ 的 "start_frame_for_prediction" 为分界
  - 分界之前的帧作为 context（模型"看到"）
  - 分界之后的帧作为 prediction target（模型"预测"）
  - 所有操作在 latent 空间完成

============================================================
实验流程
============================================================

1. 用 target_encoder 编码完整视频 → all_features [B, N_tokens, D]
2. 在时间维度上以 start_frame_for_prediction 为界切分:
     context = all_features[pre_boundary]   # 已知
     future  = all_features[post_boundary]  # 待预测（Ground Truth）
3. 对 future 特征加扩散噪声（与训练时一致）
4. 用 predictor(context, noisy_future, masks) → predicted_future
5. 计算: CosineSimilarity(predicted_future, clean_future)
6. 按 temporal position 汇总 → 相似度衰减曲线

============================================================
四个评估维度
============================================================

- Predictive Fidelity:   帧级 Cosine Similarity（越高越好）
- Temporal Stability:   相似度随 rollout 深度的衰减曲线
- Dynamics Consistency: 不同物理属性的预测难度对比
- Long-horizon Dynamics: 长距离 rollout 后是否 collapse

============================================================
如何启动
============================================================

  本地: python -m evals.main --fname configs/evals/physion_rollout.yaml --devices cuda:0
  SLURM: sbatch scripts/slurm/run_physion_rollout.sh
"""

import os
import sys
import pickle
import logging
import pprint
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import src.models.vision_transformer as vit
from src.models.predictor import VisionTransformerPredictor
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import AverageMeter, CSVLogger
from src.datasets.physion_dataset import make_physion_dataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

# 物理属性
PHYSION_PROPERTIES = ['mass', 'friction', 'elasticity', 'deformability']


def main(args_eval, resume_preempt=False):
    """Rollout 评估主函数"""

    # ===== 参数提取 =====
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

    args_data = args_eval.get('data')
    data_path = [args_data.get('dataset')]
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_data.get('frame_step', 4)
    eval_duration = args_data.get('clip_duration', None)
    data_root = args_data.get('data_root', None)  # 用于读取 .pkl 获取 start_frame

    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size', 1)  # rollout 逐视频处理
    use_bfloat16 = args_opt.get('use_bfloat16', False)

    properties_to_eval = args_eval.get('properties', None)
    if properties_to_eval is None:
        properties_to_eval = PHYSION_PROPERTIES

    eval_tag = args_eval.get('tag', None)
    rollout_steps = args_eval.get('rollout_steps', None)  # None = 全部预测帧
    noise_beta = args_eval.get('noise_beta', [0.5, 1.0])

    # ===== 分布式初始化 =====
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

    # ===== 日志和输出路径 =====
    folder = os.path.join(pretrain_folder, 'physion_rollout/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)

    # ===== 加载模型 =====
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    logger.info(f'Checkpoint keys: {list(checkpoint.keys())}')

    # 目标编码器
    encoder = vit.__dict__[model_name](
        img_size=resolution,
        patch_size=patch_size,
        num_frames=pretrain_frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    ).to(device).float()
    encoder = _load_weights(encoder, checkpoint, checkpoint_key)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 预测器（如果 checkpoint 中有）
    predictor = None
    pred_available = 'predictor' in checkpoint
    if pred_available:
        pred_state = _clean_state_dict(checkpoint['predictor'])
        # 从 state_dict 推断参数
        pred_embed_dim = pred_state['predictor_proj.weight'].shape[0]
        pred_depth = sum(1 for k in pred_state if k.startswith('predictor_blocks') and 'attn.proj.weight' in k)

        predictor = VisionTransformerPredictor(
            img_size=resolution,
            patch_size=patch_size,
            num_frames=pretrain_frames_per_clip,
            tubelet_size=tubelet_size,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=encoder.num_heads,
        ).to(device).float()
        predictor.load_state_dict(pred_state, strict=False)
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad = False
        logger.info(f'Predictor loaded: depth={pred_depth}, embed={pred_embed_dim}')
    else:
        logger.info('No predictor in checkpoint — using latent consistency mode')

    # ===== 评估循环 =====
    all_results = {}
    for prop in properties_to_eval:
        logger.info(f'\n{"="*60}')
        logger.info(f'Rollout evaluation: {prop}')
        logger.info(f'{"="*60}')

        results = evaluate_rollout(
            device=device,
            encoder=encoder,
            predictor=predictor,
            data_path=data_path,
            data_root=data_root,
            property_name=prop,
            resolution=resolution,
            frames_per_clip=eval_frames_per_clip,
            frame_step=eval_frame_step,
            eval_duration=eval_duration,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            use_bfloat16=use_bfloat16,
            tubelet_size=tubelet_size,
            rollout_steps=rollout_steps,
            noise_beta=noise_beta,
            folder=folder,
            tag=tag,
        )
        all_results[prop] = results

    # ===== 汇总输出 =====
    if rank == 0:
        _log_rollout_results(folder, tag, all_results, pred_available)


def evaluate_rollout(
    device, encoder, predictor, data_path, data_root, property_name,
    resolution, frames_per_clip, frame_step, eval_duration, batch_size,
    world_size, rank, use_bfloat16, tubelet_size,
    rollout_steps, noise_beta, folder, tag,
):
    """
    对一种物理属性执行 rollout 评估。

    流程：
    1. 加载视频
    2. 读取 start_frame_for_prediction
    3. 编码完整视频
    4. 切分 context / future
    5. 用 predictor 预测 future latent
    6. 计算 Cosine Similarity
    """

    # ---- 数据加载器 ----
    from evals.physion_attentive_probe.utils import make_physion_transforms

    transform = make_physion_transforms(
        training=False,
        crop_size=resolution,
        num_views_per_clip=1,
    )

    data_loader, _ = make_physion_dataset(
        data_paths=data_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=1,
        random_clip_sampling=False,
        duration=eval_duration,
        transform=transform,
        shared_transform=None,
        rank=rank,
        world_size=world_size,
        num_workers=2,
        pin_mem=True,
        drop_last=False,
        properties=[property_name],
        return_property_label=False,
    )

    # ---- 预计算空间 token 数 ----
    grid_size = resolution // encoder.patch_embed.patch_size
    n_spatial = (grid_size[0] if isinstance(grid_size, tuple) else grid_size) ** 2
    n_temporal = frames_per_clip // tubelet_size
    n_total = n_temporal * n_spatial

    logger.info(f'Tokens: temporal={n_temporal}, spatial={n_spatial}, total={n_total}')

    # ---- 逐视频评估 ----
    all_per_frame_sims = []  # 收集所有视频的逐帧相似度
    num_videos = 0

    # 找到数据根目录用于读取 .pkl
    if data_root is None:
        data_root = os.path.dirname(os.path.dirname(data_path[0]))
    pkl_base = data_root

    for itr, data in enumerate(data_loader):
        # 解包视频数据
        clips = data[0]
        if isinstance(clips, list):
            while isinstance(clips, list) and len(clips) > 0:
                clips = clips[0]

        # 跳过 batch_size > 1 的样本
        B = clips.shape[0]
        for b in range(B):
            video = clips[b:b+1].to(device)

            # 读取 start_frame_for_prediction
            start_frame = _get_start_frame_for_prediction(
                data_loader.dataset, itr * batch_size + b, property_name, pkl_base
            )

            if start_frame is None:
                # 默认：中点作为分界
                start_frame = frames_per_clip // 2

            t_boundary = min(start_frame // tubelet_size, n_temporal - 2)
            ctx_tokens = t_boundary * n_spatial
            tgt_tokens = n_total - ctx_tokens

            if tgt_tokens < n_spatial:  # 至少留一个时间 step
                continue

            # ---- 编码 ----
            model_dtype = next(encoder.parameters()).dtype
            video = video.to(dtype=model_dtype)

            with torch.no_grad():
                full_features = encoder(video)  # [1, N_total, D]

            # ---- 切分 context / future ----
            ctx_indices = torch.arange(ctx_tokens, device=device)
            tgt_indices = torch.arange(ctx_tokens, n_total, device=device)

            ctx_feat = full_features[:, ctx_indices, :]   # [1, N_ctx, D]
            tgt_feat = full_features[:, tgt_indices, :]   # [1, N_tgt, D]

            # ---- 预测 future latent ----
            if predictor is not None:
                # 扩散噪声
                tgt_noisy = predictor.diffusion(tgt_feat, noise_beta=tuple(noise_beta))

                # 构造 mask（predictor 需要 list 格式）
                mask_ctx = ctx_indices.unsqueeze(0).repeat(1, 1)
                mask_tgt = tgt_indices.unsqueeze(0).repeat(1, 1)

                with torch.no_grad():
                    pred = predictor(ctx_feat, tgt_noisy,
                                     [mask_ctx], [mask_tgt])
                # Cosine similarity
                sim = F.cosine_similarity(pred[0], tgt_feat[0], dim=-1)  # [N_tgt]
            else:
                # 无 predictor：测量 latent 一致性
                # 比较 encoder 和 target_encoder 的特征（这里用同一 encoder）
                # 看不同 temporal position 的特征是否有结构性差异
                sim = F.cosine_similarity(
                    full_features[0, ctx_tokens:, :],
                    full_features[0, :tgt_tokens, :] + 0.0,  # identity check
                    dim=-1
                )[:tgt_tokens]

            # ---- 按 temporal position 聚合 ----
            sim_np = sim.cpu().numpy()
            per_frame_sim = sim_np.reshape(-1, n_spatial).mean(axis=1)  # [T_future]

            all_per_frame_sims.append(per_frame_sim)
            num_videos += 1

            if (num_videos) % 20 == 0:
                avg_sim = np.mean(per_frame_sim)
                logger.info(
                    f'  [{property_name}] video {num_videos}: '
                    f'avg_sim={avg_sim:.4f}, start_frame={start_frame}'
                )

    # ---- 汇总统计 ----
    if len(all_per_frame_sims) == 0:
        logger.warning(f'No valid videos for {property_name}')
        return {'property': property_name, 'mean_sim': 0.0, 'per_frame': []}

    # 补齐到相同长度（取最短的）
    min_len = min(len(s) for s in all_per_frame_sims)
    aligned = np.array([s[:min_len] for s in all_per_frame_sims])

    mean_per_frame = aligned.mean(axis=0)    # [T_future]
    std_per_frame = aligned.std(axis=0)      # [T_future]
    overall_mean = aligned.mean()

    logger.info(
        f'[{property_name}] Rollout summary: '
        f'mean_cosine={overall_mean:.4f}, '
        f'first_frame_sim={mean_per_frame[0]:.4f}, '
        f'last_frame_sim={mean_per_frame[-1]:.4f}, '
        f'n_videos={num_videos}'
    )

    # 保存逐帧曲线
    curve_path = os.path.join(
        folder, f'{tag}_{property_name}_rollout_curve.npz'
    )
    np.savez(curve_path,
             mean=mean_per_frame, std=std_per_frame,
             n_videos=num_videos)

    return {
        'property': property_name,
        'mean_sim': float(overall_mean),
        'first_sim': float(mean_per_frame[0]),
        'last_sim': float(mean_per_frame[-1]),
        'decay': float(mean_per_frame[0] - mean_per_frame[-1]),
        'n_videos': num_videos,
        'per_frame_mean': mean_per_frame.tolist(),
    }


def _load_weights(model, checkpoint, key):
    """加载预训练权重"""
    try:
        state_dict = checkpoint[key]
    except Exception:
        state_dict = checkpoint.get('encoder', checkpoint.get('model', {}))
    state_dict = _clean_state_dict(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f'Loaded {key} from checkpoint: {msg}')
    return model


def _clean_state_dict(state_dict):
    """去除 module./backbone. 前缀"""
    sd = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('backbone.', '')
        sd[k] = v
    return sd


def _get_start_frame_for_prediction(dataset, idx, property_name, data_root):
    """
    从对应的 .pkl 文件中读取 start_frame_for_prediction。

    根据 CSV 中的视频路径，在同目录下找同编号的 .pkl 文件。
    """
    try:
        video_path = dataset.samples[idx]
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        # 从 0000_img.mp4 → 0000.pkl
        trial_num = video_name.split('_')[0]
        pkl_path = os.path.join(video_dir, f'{trial_num}.pkl')

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)

            # 尝试多种位置
            for static_key in ['static', 'static0']:
                if static_key in metadata:
                    s = metadata[static_key]
                    if 'start_frame_for_prediction' in s:
                        return int(s['start_frame_for_prediction'])

            # fallback: 在 frames 的第一帧找
            if 'frames' in metadata:
                first_key = sorted(metadata['frames'].keys())[0]
                if 'labels' in metadata['frames'][first_key]:
                    return None  # 没有全局 start_frame
    except Exception:
        pass

    return None  # 默认会在调用处设为 frames_per_clip // 2


def _log_rollout_results(folder, tag, all_results, has_predictor):
    """输出 rollout 结果"""

    mode = 'predictor' if has_predictor else 'latent_consistency'
    results_path = os.path.join(folder, f'{tag}_{mode}_rollout_results.txt')

    lines = []
    lines.append('=' * 60)
    lines.append(f'V-JEPA Physion++ Rollout Results [{mode}]')
    lines.append('=' * 60)
    lines.append(f'  {"Property":20s}  MeanSim  First   Last   Decay  ')
    lines.append('  ' + '-' * 52)

    for prop in PHYSION_PROPERTIES:
        if prop not in all_results:
            continue
        r = all_results[prop]
        lines.append(
            f'  {prop:20s}  {r["mean_sim"]:.4f}  '
            f'{r["first_sim"]:.4f}  {r["last_sim"]:.4f}  {r["decay"]:.4f}'
        )

    lines.append('=' * 60)

    for line in lines:
        logger.info(line)

    with open(results_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f'Results saved to: {results_path}')
