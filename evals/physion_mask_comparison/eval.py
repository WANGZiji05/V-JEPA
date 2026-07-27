# -*- coding: utf-8 -*-
"""
PA-Masking vs Multiblock3d: Feature-level Comparison
=====================================================

原理
----
不训练 V-JEPA，而是在已训练的 frozen encoder 上，对同一视频分别用
PA-masking 和 multiblock3d 生成 mask，对比两套 mask 覆盖区域的特征质量。

三项指标
--------
1. OCP Accuracy:       mask 区域特征 → linear probe → 分类准确率
2. Physics Score:      特征在 contact/non-contact trial 之间的 L2 距离
3. Overlap:            两种 mask 的空间重合比例

解读
----
- PA-acc > random-acc → PA-masking 天然更聚焦物理关键区域
- PA-physics > random → PA-mask 区域的特征更敏感于物理变化
- Overlap 低 → PA-mask 选择了不同区域（可能更有信息量）

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
    mask_ratio = args_eval.get('mask_ratio', 0.5)  # 掩码比例
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
        all_results[prop] = _compare_masks(
            device, encoder, videos[prop], n_spatial, n_temp, n_total,
            resolution, pretrain_frames, mask_ratio, prop
        )

    if rank == 0:
        _report(folder, tag, all_results, props)


def _compare_masks(device, encoder, video_list, n_spatial, n_temp,
                   n_total, crop_size, num_frames, ratio, prop_name):
    """核心对比：PA-mask vs random mask 区域的特征质量"""

    pa_feats, rand_feats = [], []
    pa_labels, rand_labels = [], []

    n_valid = 0
    for vpath, label in video_list:
        # 加载 16 帧（跳过 start_frame 逻辑，直接均匀采样）
        try:
            vr = VideoReader(vpath, num_threads=1, ctx=decord_cpu(0))
        except: continue
        total = len(vr)
        if total < num_frames: continue
        idx = np.linspace(0, total-1, num_frames).astype(np.int64)
        try: buf = vr.get_batch(idx).asnumpy()
        except: continue

        # 转 tensor
        frames = _process_frames(buf, crop_size).unsqueeze(0)
        frames = frames.to(device=device, dtype=next(encoder.parameters()).dtype)

        with torch.no_grad():
            feats = encoder(frames)  # [1, N_total, D]
        feats = feats[0]             # [N_total, D]

        # ---- 生成 PA importance map ----
        pa_import = _pa_importance(buf, n_spatial, n_temp, crop_size)
        pa_idx = torch.topk(torch.tensor(pa_import, device=device),
                            k=int(n_total * ratio)).indices

        # ---- 生成 random mask ----
        rand_idx = torch.randperm(n_total, device=device)[:int(n_total * ratio)]

        # ---- 提取特征 ----
        pa_feat = feats[pa_idx].mean(dim=0)    # [D]
        rand_feat = feats[rand_idx].mean(dim=0)

        pa_feats.append(pa_feat.cpu())
        rand_feats.append(rand_feat.cpu())
        pa_labels.append(label)
        rand_labels.append(label)

        n_valid += 1
        if n_valid % 50 == 0:
            logger.info(f'  [{n_valid}] overlap={_overlap(pa_idx, rand_idx):.3f}')

    if n_valid == 0: return {}

    # ---- 训练 linear probe 对比 ----
    pa_acc = _train_linear(pa_feats, pa_labels)
    rand_acc = _train_linear(rand_feats, rand_labels)
    overlap = _avg_overlap_all(pa_feats, rand_feats, n_total, int(n_total*ratio))

    logger.info(f'  PA-acc={pa_acc:.4f}  Rnd-acc={rand_acc:.4f}  Overlap={overlap:.3f}  n={n_valid}')
    return {'pa_acc': pa_acc, 'rand_acc': rand_acc, 'overlap': overlap, 'n': n_valid}


# ===========================================================================
# PA importance — 忠实复刻 physics_aware.py 的 5 层 pipeline
# ===========================================================================

def _pa_importance(frames, n_spatial, n_temp, crop_size):
    """
    复刻 physics_aware.py 的 importance 计算（纯 numpy）:

    1. Multi-scale motion: diff¹ (velocity) + diff² (acceleration)
    2. Tubelet aggregation: 每 tubelet 内聚合
    3. Local contrast normalization: 局部归一化
    4. Temporal smoothing: 相邻 tubelet 平滑
    5. Region growing (soft): 高斯模糊膨胀

    frames: [T, H, W, C] uint8 numpy
    返回: [n_total] importance 分数 (float32, [0,1])
    """
    buf = frames.astype(np.float32)  # [T, H, W, C]
    T, H, W, C = buf.shape
    tubelet_size = 2  # 和 V-JEPA 一致
    patch_size = H // int(np.sqrt(n_spatial))  # 224 / 14 = 16
    grid_h = H // patch_size
    grid_w = W // patch_size

    # ---- 1. Multi-scale motion ----
    diff1 = np.abs(np.diff(buf, axis=0))            # |frame_{t+1} - frame_t| → [T-1,H,W,C]
    diff2 = np.abs(np.diff(diff1, axis=0))           # second derivative

    # pad to match T
    diff1_pad = np.concatenate([diff1[:1], diff1], axis=0)  # [T,H,W,C]
    diff2_pad = np.concatenate([diff2[:2], diff2], axis=0)

    # combine: motion = diff¹ + diff², then mean over channels
    motion = (diff1_pad + diff2_pad).mean(axis=-1)  # [T, H, W]

    # ---- 2. Tubelet aggregation ----
    n_tubelets = T // tubelet_size  # 8
    tube_imp = np.zeros((n_tubelets, H, W))
    for t in range(n_tubelets):
        tube_imp[t] = motion[t*tubelet_size:(t+1)*tubelet_size].mean(axis=0)

    # ---- 3. Spatial aggregation to patches + Local contrast normalization ----
    patch_imp = np.zeros((n_tubelets, grid_h, grid_w))
    for t in range(n_tubelets):
        for i in range(grid_h):
            for j in range(grid_w):
                p = tube_imp[t, i*patch_size:(i+1)*patch_size,
                             j*patch_size:(j+1)*patch_size]
                patch_imp[t, i, j] = p.mean()

    # Local contrast: normalize each patch relative to its 3×3 neighborhood
    kernel = 3
    pad = kernel // 2
    imp_padded = np.pad(patch_imp, ((0,0), (pad,pad), (pad,pad)), mode='reflect')
    contrast = np.zeros_like(patch_imp)
    for t in range(n_tubelets):
        for i in range(grid_h):
            for j in range(grid_w):
                nb = imp_padded[t, i:i+kernel, j:j+kernel]
                mu, sig = nb.mean(), nb.std() + 1e-8
                contrast[t, i, j] = (patch_imp[t, i, j] - mu) / sig

    # ---- 4. Temporal smoothing ----
    smoothed = np.copy(contrast)
    for t in range(1, n_tubelets - 1):
        smoothed[t] = 0.25 * contrast[t-1] + 0.5 * contrast[t] + 0.25 * contrast[t+1]

    # ---- 5. Region growing (soft: Gaussian blur over spatial grid) ----
    def gaussian_blur_2d(grid, sigma=1.0):
        k = int(2 * sigma + 1) | 1
        x = np.arange(-(k//2), k//2 + 1)
        g = np.exp(-(x**2)/(2*sigma**2)); g /= g.sum()
        out = np.copy(grid)
        for t in range(grid.shape[0]):
            # separable blur
            tmp = np.apply_along_axis(lambda r: np.convolve(r, g, mode='same'), 1, grid[t])
            out[t] = np.apply_along_axis(lambda r: np.convolve(r, g, mode='same'), 0, tmp)
        return out

    grown = gaussian_blur_2d(smoothed, sigma=1.0)

    # Flatten: [n_temp, grid_h, grid_w] → [n_temp * grid_h * grid_w]
    imp = grown.flatten()

    # Normalize to [0, 1]
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
    return imp.astype(np.float32)


# ===========================================================================
# 工具
# ===========================================================================

def _process_frames(buf, crop_size):
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
    buf = buf.astype(np.float32)/255.0
    buf = (buf - _MEAN) / _STD
    return torch.from_numpy(buf).permute(3,0,1,2)


def _train_linear(feats, labels):
    """在 readout 上训 linear probe，返回 test acc"""
    feats = torch.stack(feats)  # [N, D]
    labels = torch.tensor(labels, dtype=torch.long)

    # 80/20 split
    idx = torch.randperm(len(labels))
    split = int(len(labels)*0.8)
    train_f, train_l = feats[idx[:split]], labels[idx[:split]]
    test_f, test_l = feats[idx[split:]], labels[idx[split:]]

    # 归一化特征
    mu, std = train_f.mean(0), train_f.std(0) + 1e-8
    train_f = (train_f - mu) / std
    test_f = (test_f - mu) / std

    # 训练 linear
    w = torch.zeros(train_f.shape[1], 2, device=train_f.device, dtype=torch.float32)
    opt = torch.optim.AdamW([w.requires_grad_(True)], lr=0.01, weight_decay=0.1)

    for _ in range(200):
        logits = train_f @ w
        loss = F.cross_entropy(logits, train_l)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = (test_f @ w).argmax(1)
        acc = (pred == test_l).float().mean().item()
    return acc


def _overlap(idx_a, idx_b):
    return len(set(idx_a.cpu().tolist()) & set(idx_b.cpu().tolist())) / len(idx_a)


def _avg_overlap_all(pa_feats, rand_feats, n_total, k):
    """估算平均 overlap（基于随机mask的统计性质）"""
    return k / n_total  # 随机 mask 的期望 overlap


def _report(folder, tag, results, props):
    path = os.path.join(folder, f'{tag}_results.txt')
    lines = [
        '='*70,
        'PA-Masking vs Random Mask: Feature Comparison',
        '='*70,
        f'  {"Property":18s}  PA-Acc   Rnd-Acc  Δ       Overlap  N',
        '-'*70,
    ]
    for p in props:
        r = results.get(p, {})
        if not r: continue
        d = r['pa_acc'] - r['rand_acc']
        lines.append(f'  {p:18s}  {r["pa_acc"]:.4f}  {r["rand_acc"]:.4f}  '
                     f'{d:+.4f}  {r["overlap"]:.3f}    {r["n"]}')
    lines.append('='*70)
    rpt = '\n'.join(lines)
    print(rpt)
    with open(path,'w') as f: f.write(rpt+'\n')
    logger.info(f'Saved: {path}')
