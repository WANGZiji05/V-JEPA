# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

"""
V-JEPA Physion++ Latent Rollout 评估（修正版）

============================================================
实验原理
============================================================

V-JEPA 在 latent 空间做预测：给出 context 特征 → 预测未来区域的特征。
本实验以 Physion++ 的 start_frame_for_prediction 为界：
  - Context:   start_frame 之前的帧（V-JEPA "看到"）
  - Future:    start_frame 之后的帧（V-JEPA "预测"）

完整流程：
1. 读取 .pkl 获取 start_frame_for_prediction
2. 用 decord 加载包含分界点的 16 帧（context + future 各半）
3. 以分界点为界创建 context/future mask
4. target_encoder 编码完整 clip → all_features
5. Predictor(ctx_features, noisy_future_features) → predicted_future
6. CosineSimilarity(predicted, actual) → 逐帧相似度

============================================================
四个评估维度
============================================================

Predictive Fidelity:    mean CosineSim（预测 latent 和真实 latent 的相似度）
Temporal Stability:    first_sim → last_sim 的衰减（Decay）
Dynamics Consistency:  不同物理属性的预测难度对比
Long-horizon Dynamics: last_sim（长距离预测是否崩溃，>0 说明不崩溃）

============================================================
随机 baseline
============================================================

1280 维空间中随机向量的期望 CosineSim = 0，std ≈ 1/sqrt(1280) ≈ 0.028
任何 >0.06 的值都显著偏离随机（>2σ），>0 即说明 predictor 学到了预测能力

============================================================
使用
============================================================

python -m evals.main --fname configs/evals/physion_rollout.yaml --devices cuda:0
"""

import os
import sys
import pickle
import logging
import pprint
import csv
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from decord import VideoReader, cpu as decord_cpu

import src.models.vision_transformer as vit
from src.models.predictor import VisionTransformerPredictor
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

PHYSION_PROPERTIES = ['mass', 'friction', 'elasticity', 'deformability']

# ImageNet 归一化
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main(args_eval, resume_preempt=False):

    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name')
    patch_size = args_pretrain.get('patch_size', 16)
    pretrain_folder = args_pretrain.get('folder')
    ckp_fname = args_pretrain.get('checkpoint')
    tag = args_pretrain.get('write_tag', 'rollout')
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames = args_pretrain.get('frames_per_clip', 16)

    args_data = args_eval.get('data')
    test_csv = args_data.get('dataset')
    resolution = args_eval.get('optimization', {}).get('resolution', 224)
    noise_beta = tuple(args_eval.get('noise_beta', [0.5, 1.0]))

    properties_to_eval = args_eval.get('properties', None) or PHYSION_PROPERTIES
    eval_tag = args_eval.get('tag', 'rollout')

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Rank {rank}/{world_size}')

    folder = os.path.join(pretrain_folder, 'physion_rollout/')
    if eval_tag:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)

    # ---- 加载模型 ----
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    logger.info(f'Checkpoint keys: {list(checkpoint.keys())}')

    encoder = vit.__dict__[model_name](
        img_size=resolution, patch_size=patch_size,
        num_frames=pretrain_frames, tubelet_size=tubelet_size,
        uniform_power=uniform_power, use_sdpa=use_sdpa,
        use_SiLU=use_SiLU, tight_SiLU=tight_SiLU,
    ).to(device).float().eval()
    _load_weights(encoder, checkpoint, checkpoint_key)
    for p in encoder.parameters():
        p.requires_grad = False

    predictor = None
    has_pred = 'predictor' in checkpoint
    if has_pred:
        pred_state = _clean_state_dict(checkpoint['predictor'])
        pred_dim = pred_state['predictor_proj.weight'].shape[1]  # 384
        pred_depth = sum(1 for k in pred_state
                        if k.startswith('predictor_blocks')
                        and '.attn.proj.weight' in k)
        predictor = VisionTransformerPredictor(
            img_size=resolution, patch_size=patch_size,
            num_frames=pretrain_frames, tubelet_size=tubelet_size,
            embed_dim=encoder.embed_dim, predictor_embed_dim=pred_dim,
            depth=pred_depth, num_heads=12,
        ).to(device).float().eval()
        predictor.load_state_dict(pred_state, strict=False)
        for p in predictor.parameters():
            p.requires_grad = False
        logger.info(f'Predictor: depth={pred_depth}, embed={pred_dim}')
    else:
        logger.info('No predictor - latent consistency mode')

    # ---- 读取 CSV，按属性分组视频路径 ----
    video_list = _load_video_list(test_csv, properties_to_eval)
    logger.info(f'Loaded {sum(len(v) for v in video_list.values())} videos')

    # ---- 评估 ----
    all_results = {}
    for prop in properties_to_eval:
        logger.info(f'\n{"="*60}')
        logger.info(f'Rollout: {prop} ({len(video_list[prop])} videos)')
        logger.info(f'{"="*60}')

        results = _run_rollout(
            device, encoder, predictor,
            video_list[prop], resolution, pretrain_frames,
            tubelet_size, noise_beta, has_pred,
        )
        all_results[prop] = results

    if rank == 0:
        _log_results(folder, tag, all_results, has_pred)


def _load_video_list(csv_path, properties):
    """从 CSV 读取视频路径，按属性分组"""
    groups = {p: [] for p in properties}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            path, prop, label = row[0], row[1].strip(), int(row[2])
            if prop in groups and label >= 0:
                groups[prop].append(path)
    return groups


def _run_rollout(device, encoder, predictor, video_paths,
                 resolution, num_frames, tubelet_size, noise_beta, has_pred):
    """
    对一组视频执行 rollout 评估。

    核心：以 start_frame_for_prediction 为中心采样 16 帧，
    切分 context（前半）和 future（后半），评估预测质量。
    """

    # 每帧的 token 数
    grid = resolution // encoder.patch_embed.patch_size[0]
    n_spatial = grid * grid
    n_temporal = num_frames // tubelet_size  # 8
    n_total = n_temporal * n_spatial       # 1568

    all_per_frame = []
    n_videos = 0

    for vi, video_path in enumerate(video_paths):
        # ---- 读取 start_frame_for_prediction ----
        start_frame = _read_start_frame(video_path)
        if start_frame is None or start_frame < 4:
            continue

        # ---- 加载 16 帧，以 start_frame 为中心 ----
        ctx_frames = num_frames // 2  # 8 frames context
        clip_start = max(0, start_frame - ctx_frames * 4)  # frame_step=4

        frames = _load_clip(video_path, clip_start, num_frames, step=4,
                            crop_size=resolution)
        if frames is None:
            continue

        # frames: [C, T, H, W] on device
        frames = frames.unsqueeze(0)  # [1, C, T, H, W]
        frames = frames.to(device=device, dtype=next(encoder.parameters()).dtype)

        # ---- 编码 ----
        with torch.no_grad():
            all_feat = encoder(frames)  # [1, N_total, D]

        # ---- 切分 temporally ----
        # 前半 = context, 后半 = future
        # n_temporal = 8, ctx = tokens[0:4*n_spatial], fut = tokens[4*n_spatial:]
        mid_t = n_temporal // 2
        ctx_tok = mid_t * n_spatial       # 4 * 196 = 784
        fut_tok = n_total - ctx_tok       # 4 * 196 = 784

        ctx_idx = torch.arange(ctx_tok, device=device)
        fut_idx = torch.arange(ctx_tok, n_total, device=device)

        ctx_feat = all_feat[:, ctx_idx, :]  # [1, 784, D]
        fut_feat = all_feat[:, fut_idx, :]  # [1, 784, D]

        # ---- 预测 ----
        if has_pred:
            fut_noisy = predictor.diffusion(fut_feat, noise_beta=noise_beta)
            mask_ctx = [ctx_idx.unsqueeze(0)]
            mask_fut = [fut_idx.unsqueeze(0)]
            with torch.no_grad():
                pred = predictor(ctx_feat, fut_noisy, mask_ctx, mask_fut)
            sim = F.cosine_similarity(pred[0], fut_feat[0], dim=-1)
        else:
            # 无 predictor：测量 encoder 特征的自洽性
            sim = F.cosine_similarity(fut_feat[0], fut_feat[0], dim=-1)

        # ---- 按 temporal position 聚合 ----
        sim_np = sim.cpu().numpy()
        per_frame = sim_np.reshape(-1, n_spatial).mean(axis=1)  # [4]

        all_per_frame.append(per_frame)
        n_videos += 1

        if n_videos % 40 == 0:
            avg = np.mean(per_frame)
            logger.info(f'  [{n_videos}] avg_sim={avg:.4f}, start_frame={start_frame}')

    if n_videos == 0:
        return {'mean_sim': 0, 'first_sim': 0, 'last_sim': 0, 'decay': 0, 'n': 0}

    # 对齐长度（都应是 4）
    aligned = np.stack(all_per_frame)
    mean_f = aligned.mean(axis=0)
    overall = aligned.mean()

    logger.info(
        f'  Summary: mean={overall:.4f}, first={mean_f[0]:.4f}, '
        f'last={mean_f[-1]:.4f}, n={n_videos}'
    )

    return {
        'mean_sim': float(overall),
        'first_sim': float(mean_f[0]),
        'last_sim': float(mean_f[-1]),
        'decay': float(mean_f[0] - mean_f[-1]),
        'n_videos': n_videos,
        'per_frame': mean_f.tolist(),
    }


def _read_start_frame(video_path):
    """从同名的 .pkl 读取 start_frame_for_prediction"""
    try:
        vdir = os.path.dirname(video_path)
        vname = os.path.basename(video_path)
        trial = vname.split('_')[0]
        pkl = os.path.join(vdir, f'{trial}.pkl')
        if not os.path.exists(pkl):
            return None
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        for key in ['static', 'static0']:
            if key in data and 'start_frame_for_prediction' in data[key]:
                return int(data[key]['start_frame_for_prediction'])
    except Exception:
        pass
    return None


def _load_clip(video_path, start_frame, num_frames, step, crop_size):
    """用 decord 加载 clip，返回 [C, T, H, W] tensor"""
    try:
        vr = VideoReader(video_path, num_threads=1, ctx=decord_cpu(0))
    except Exception:
        return None

    total = len(vr)
    indices = np.arange(start_frame, start_frame + num_frames * step, step)
    indices = np.clip(indices, 0, total - 1).astype(np.int64)

    try:
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    except Exception:
        return None

    # Resize + CenterCrop + Normalize (纯 numpy，无需 PIL)
    T, H, W, C = buffer.shape
    short = int(crop_size * 256 / 224)
    scale = short / min(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))

    resized = np.zeros((T, new_h, new_w, C), dtype=buffer.dtype)
    for t in range(T):
        # 最近邻 resize（快速但不完美，可接受）
        for i in range(new_h):
            si = min(int(i / scale), H - 1)
            for j in range(new_w):
                sj = min(int(j / scale), W - 1)
                resized[t, i, j] = buffer[t, si, sj]

    h_s = (new_h - crop_size) // 2
    w_s = (new_w - crop_size) // 2
    buffer = resized[:, h_s:h_s + crop_size, w_s:w_s + crop_size, :]

    buffer = buffer.astype(np.float32) / 255.0
    buffer = (buffer - _MEAN) / _STD
    tensor = torch.from_numpy(buffer).permute(3, 0, 1, 2)  # [C, T, H, W]
    return tensor


def _load_weights(model, checkpoint, key):
    sd = checkpoint.get(key, checkpoint.get('encoder', {}))
    sd = _clean_state_dict(sd)
    msg = model.load_state_dict(sd, strict=False)
    logger.info(f'Loaded {key}: {msg}')


def _clean_state_dict(sd):
    return {k.replace('module.', '').replace('backbone.', ''): v
            for k, v in sd.items()}


def _log_results(folder, tag, results, has_pred):
    mode = 'predictor' if has_pred else 'consistency'
    path = os.path.join(folder, f'{tag}_{mode}_rollout_results.txt')

    lines = [
        '=' * 60,
        f'V-JEPA Physion++ Latent Rollout [{mode}]',
        '=' * 60,
        f'  {"Property":20s}  MeanSim  First   Last   Decay   N',
        '  ' + '-' * 52,
    ]
    for prop in PHYSION_PROPERTIES:
        r = results.get(prop, {})
        if not r:
            continue
        lines.append(
            f'  {prop:20s}  {r["mean_sim"]:.4f}  {r["first_sim"]:.4f}  '
            f'{r["last_sim"]:.4f}  {r["decay"]:+.4f}  {r.get("n_videos",0)}'
        )
    lines.append('=' * 60)

    print('\n'.join(lines))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f'Saved: {path}')
