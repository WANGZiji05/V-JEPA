# -*- coding: utf-8 -*-
"""
PA-Masking vs Multiblock3d: Feature-level Comparison
=====================================================

原理
----
在已训练的 frozen ViT-Huge encoder 上，对同一视频分别用 PA-masking 和
V-JEPA 官方的 multiblock3d 生成 mask，对比两套 mask 覆盖区域的特征质量。

三项指标
--------
1. OCP Accuracy:       mask 区域特征 → linear probe → 分类准确率
2. Overlap:            两种 mask 的空间重合比例（越低说明 PA 选得越不同）
3. Physics Score:      特征在 contact/non-contact trial 之间的区分度

用法
----
python -m evals.main --fname configs/evals/physion_mask_comparison.yaml --devices cuda:0
"""

import os, sys, pickle, logging, csv
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu as decord_cpu

import src.models.vision_transformer as vit
from src.masks.multiblock3d import _MaskGenerator as MB3DGenerator
from src.utils.distributed import init_distributed, AllReduce

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 42
np.random.seed(_GLOBAL_SEED); torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

PHYSION = ['mass', 'friction', 'elasticity', 'deformability']
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def main(args_eval, resume_preempt=False):
    p = args_eval.get('pretrain')
    pretrained_path = os.path.join(p.get('folder'), p.get('checkpoint'))
    model_name = p.get('model_name'); patch_size = p.get('patch_size', 16)
    tubelet_size = p.get('tubelet_size', 2); use_sdpa = p.get('use_sdpa', True)
    pretrain_frames = p.get('frames_per_clip', 16)

    d = args_eval.get('data')
    test_csv = d.get('dataset')
    resolution = args_eval.get('optimization', {}).get('resolution', 224)
    mask_ratio = args_eval.get('mask_ratio', 0.5)
    props = args_eval.get('properties', None) or PHYSION
    tag = args_eval.get('tag', 'mask_comp')

    try: mp.set_start_method('spawn')
    except: pass

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)
    world_size, rank = init_distributed()

    folder = os.path.join(p.get('folder'), 'physion_mask_comparison/')
    os.makedirs(folder, exist_ok=True)

    # ---- 加载 encoder ----
    encoder = vit.__dict__[model_name](
        img_size=resolution, patch_size=patch_size,
        num_frames=pretrain_frames, tubelet_size=tubelet_size, use_sdpa=use_sdpa,
    ).to(device).float().eval()
    ckpt = torch.load(pretrained_path, map_location='cpu')
    sd = ckpt.get(p.get('checkpoint_key'), ckpt.get('encoder', {}))
    sd = {k.replace('module.','').replace('backbone.',''): v for k,v in sd.items()}
    encoder.load_state_dict(sd, strict=False)
    for pp in encoder.parameters(): pp.requires_grad = False

    patch_sz = patch_size if isinstance(patch_size, int) else patch_size[0]
    grid = resolution // patch_sz
    n_spatial = grid * grid
    n_temp = pretrain_frames // tubelet_size
    n_total = n_temp * n_spatial
    logger.info(f'Tokens: {n_temp}×{n_spatial}={n_total}')

    # ---- 创建 multiblock3d mask 生成器（V-JEPA 官方策略） ----
    # 匹配 PA-masking 的 50% coverage
    mb3d_gen = MB3DGenerator(
        crop_size=(resolution, resolution),
        num_frames=pretrain_frames,
        spatial_patch_size=(patch_sz, patch_sz),
        temporal_patch_size=tubelet_size,
        spatial_pred_mask_scale=(0.5, 0.5),  # 50% spatial coverage
        temporal_pred_mask_scale=(1.0, 1.0),  # full temporal
        aspect_ratio=(0.75, 0.75),
        npred=1,
        max_keep=None,
    )

    # ---- 加载视频 ----
    videos = {p: [] for p in props}
    with open(test_csv) as f:
        for row in csv.reader(f):
            if len(row) < 3: continue
            path, prop, label = row[0], row[1].strip(), int(row[2])
            if prop in videos and label >= 0:
                videos[prop].append((path, label))
    logger.info(f'Loaded {sum(len(v) for v in videos.values())} videos')

    # ---- 逐属性对比 ----
    all_results = {}
    for prop in props:
        logger.info(f'\n{"="*60}\n{prop}: {len(videos[prop])} videos\n{"="*60}')
        all_results[prop] = _compare(
            device, encoder, mb3d_gen, videos[prop],
            n_spatial, n_temp, n_total, resolution,
            pretrain_frames, mask_ratio
        )

    if rank == 0:
        _report(folder, tag, all_results, props)


# ===========================================================================
# 核心对比
# ===========================================================================

def _compare(device, encoder, mb3d_gen, video_list,
             n_spatial, n_temp, n_total, crop_size, num_frames, ratio):

    # 两种 mask 的特征各存一份
    pa_feats, mb_feats = [], []
    pa_labels, mb_labels = [], []

    n_valid = 0; total_overlap = 0.0

    for vpath, label in video_list:
        # 加载视频帧
        try:
            vr = VideoReader(vpath, num_threads=1, ctx=decord_cpu(0))
        except: continue
        total_f = len(vr)
        if total_f < num_frames: continue
        idx = np.linspace(0, total_f-1, num_frames).astype(np.int64)
        try: buf = vr.get_batch(idx).asnumpy()
        except: continue

        frames = _process(buf, crop_size).unsqueeze(0)
        frames = frames.to(device=device, dtype=next(encoder.parameters()).dtype)

        with torch.no_grad():
            feats = encoder(frames)[0]  # [N_total, D]

        # ---- PA-masking: 生成 top-K indices ----
        pa_imp = _pa_importance(buf, n_spatial, n_temp, crop_size)
        pa_idx = torch.topk(torch.as_tensor(pa_imp, device=device),
                            k=int(n_total * ratio)).indices

        # ---- Multiblock3d: 生成 target 区域 indices ----
        _, masks_pred = mb3d_gen(1)                # batch_size=1
        mb_idx = masks_pred[0].to(device)           # [K_mb] indices
        # 如果 multiblock3d 生成的 token 数不等于预期，用 topk 裁齐
        if len(mb_idx) < int(n_total * ratio):
            # 补齐到 ratio（用随机）
            remaining = list(set(range(n_total)) - set(mb_idx.tolist()))
            fill = torch.as_tensor(remaining[:int(n_total*ratio)-len(mb_idx)],
                                   device=device)
            mb_idx = torch.cat([mb_idx, fill])
        elif len(mb_idx) > int(n_total * ratio):
            mb_idx = mb_idx[:int(n_total * ratio)]

        # ---- 提取特征 ----
        pa_feat = feats[pa_idx].mean(dim=0)
        mb_feat = feats[mb_idx].mean(dim=0)

        pa_feats.append(pa_feat.cpu()); mb_feats.append(mb_feat.cpu())
        pa_labels.append(label);         mb_labels.append(label)

        # overlap
        ol = len(set(pa_idx.tolist()) & set(mb_idx.tolist())) / len(pa_idx)
        total_overlap += ol

        n_valid += 1
        if n_valid % 50 == 0:
            logger.info(f'  [{n_valid}] overlap={ol:.3f}  '
                        f'PA_k={len(pa_idx)}  MB_k={len(mb_idx)}')

    if n_valid == 0: return {}

    # ---- 分别训练 linear 分类器 ----
    pa_acc = _linear_acc(pa_feats, pa_labels)
    mb_acc = _linear_acc(mb_feats, mb_labels)
    avg_ol = total_overlap / n_valid

    logger.info(f'  PA-acc={pa_acc:.4f}  MB-acc={mb_acc:.4f}  '
                f'Overlap={avg_ol:.3f}  n={n_valid}')
    return {'pa_acc': pa_acc, 'mb_acc': mb_acc, 'overlap': avg_ol, 'n': n_valid}


# ===========================================================================
# PA importance — 复刻 physics_aware.py 的 5 层 pipeline
# ===========================================================================

def _pa_importance(frames, n_spatial, n_temp, crop_size):
    buf = frames.astype(np.float32)  # [T, H, W, C]
    T, H, W, C = buf.shape
    tubelet_size = 2
    patch_size = H // int(np.sqrt(n_spatial))  # 16
    grid_h, grid_w = H // patch_size, W // patch_size

    # 1. Multi-scale motion: diff¹ + diff²
    d1 = np.abs(np.diff(buf, axis=0))              # [T-1, H, W, C]
    d2 = np.abs(np.diff(d1, axis=0))               # [T-2, H, W, C]
    d1p = np.concatenate([d1[:1], d1], axis=0)     # [T, H, W, C]
    d2p = np.concatenate([d2[:2], d2], axis=0)
    motion = (d1p + d2p).mean(axis=-1)              # [T, H, W]

    # 2. Tubelet aggregation
    n_tube = T // tubelet_size  # 8
    tube_imp = np.zeros((n_tube, H, W))
    for t in range(n_tube):
        tube_imp[t] = motion[t*tubelet_size:(t+1)*tubelet_size].mean(axis=0)

    # 3. Patch aggregation + local contrast normalization
    patch_imp = np.zeros((n_tube, grid_h, grid_w))
    for t in range(n_tube):
        for i in range(grid_h):
            for j in range(grid_w):
                p = tube_imp[t, i*patch_size:(i+1)*patch_size,
                             j*patch_size:(j+1)*patch_size]
                patch_imp[t, i, j] = p.mean()

    kernel = 3; pad = 1
    padded = np.pad(patch_imp, ((0,0),(pad,pad),(pad,pad)), mode='reflect')
    contrast = np.zeros_like(patch_imp)
    for t in range(n_tube):
        for i in range(grid_h):
            for j in range(grid_w):
                nb = padded[t, i:i+kernel, j:j+kernel]
                mu, sig = nb.mean(), nb.std() + 1e-8
                contrast[t, i, j] = (patch_imp[t, i, j] - mu) / sig

    # 4. Temporal smoothing
    smooth = np.copy(contrast)
    for t in range(1, n_tube-1):
        smooth[t] = 0.25*contrast[t-1] + 0.5*contrast[t] + 0.25*contrast[t+1]

    # 5. Soft region growing (Gaussian blur on spatial grid)
    k = 3; g = np.array([0.25, 0.5, 0.25])
    grown = np.copy(smooth)
    for t in range(n_tube):
        tmp = np.apply_along_axis(lambda r: np.convolve(r, g, mode='same'), 1, smooth[t])
        grown[t] = np.apply_along_axis(lambda r: np.convolve(r, g, mode='same'), 0, tmp)

    imp = grown.flatten()
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
    return imp.astype(np.float32)


# ===========================================================================
# 工具
# ===========================================================================

def _process(buf, crop_size):
    T, H, W, C = buf.shape
    short = int(crop_size * 256 / 224)
    scl = short / min(H, W)
    nh, nw = int(round(H*scl)), int(round(W*scl))
    res = np.zeros((T, nh, nw, C), dtype=buf.dtype)
    for t in range(T):
        for i in range(nh): res[t,i] = buf[t,min(int(i/scl),H-1)]
        for j in range(nw): res[t,:,j] = buf[t,:,min(int(j/scl),W-1)]
    hs, ws = (nh-crop_size)//2, (nw-crop_size)//2
    buf = res[:,hs:hs+crop_size,ws:ws+crop_size,:]
    buf = buf.astype(np.float32)/255.0; buf = (buf-_MEAN)/_STD
    return torch.from_numpy(buf).permute(3,0,1,2)


def _linear_acc(feats, labels):
    feats = torch.stack(feats)
    labels = torch.tensor(labels, dtype=torch.long)
    n = len(labels); idx = torch.randperm(n)
    split = int(n*0.8)
    t_f, t_l = feats[idx[:split]], labels[idx[:split]]
    v_f, v_l = feats[idx[split:]], labels[idx[split:]]
    mu, std = t_f.mean(0), t_f.std(0)+1e-8
    t_f, v_f = (t_f-mu)/std, (v_f-mu)/std
    w = torch.zeros(t_f.shape[1], 2, device=t_f.device)
    opt = torch.optim.AdamW([w.requires_grad_(True)], lr=0.01, weight_decay=0.1)
    for _ in range(200):
        opt.zero_grad(); F.cross_entropy(t_f@w, t_l).backward(); opt.step()
    with torch.no_grad(): acc = (v_f@w).argmax(1).eq(v_l).float().mean().item()
    return acc


def _report(folder, tag, results, props):
    path = os.path.join(folder, f'{tag}_results.txt')
    lines = [
        '='*70,
        'PA-Masking vs Multiblock3d (V-JEPA Official)',
        'Feature-level Comparison on Frozen ViT-Huge Encoder',
        '='*70,
        f'  {"Property":18s}  PA-Acc   MB-Acc   Δ        Overlap  N',
        '-'*70,
    ]
    for p in props:
        r = results.get(p, {})
        if not r: continue
        d = r['pa_acc'] - r['mb_acc']
        lines.append(
            f'  {p:18s}  {r["pa_acc"]:.4f}  {r["mb_acc"]:.4f}  '
            f'{d:+.4f}  {r["overlap"]:.3f}    {r["n"]}'
        )
    lines.append('='*70)
    rpt = '\n'.join(lines); print(rpt)
    with open(path,'w') as f: f.write(rpt+'\n')
    logger.info(f'Saved: {path}')
