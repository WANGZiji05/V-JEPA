# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
"""
V-JEPA Physion++ Latent Dynamics Evaluation
============================================================

实验设计
============================================================

[核心实验] Prediction Horizon Evaluation
  固定 context 在 start_frame_for_prediction 之前，
  逐步增加预测距离 (1 → 7 temporal tokens)，
  测量预测相似度随 horizon 的衰减曲线。

[辅助实验] Context Ablation
  对每个 horizon：
    With ctx:    context_features + noisy(future) → sim_A
    Without ctx: zeros            + noisy(future) → sim_B
    Δ = sim_A - sim_B = "context 对预测的贡献"

[分组分析] Per-Property
  分别对 mass / friction / elasticity / deformability 评估。

============================================================
为什么这样设计
============================================================
V-JEPA 的 predictor 不是自回归的。
它同时接收 context tokens + target tokens（diffusion 加噪）作为输入。
所有 target tokens 并行预测，没有反馈回路。
因此不能做 recursive rollout。

正确评估 latent dynamics 的方式：
"预测越远的未来（更多 target tokens），或给越少的 context，
 模型是否仍能正确预测未来 latent？"

Horizon 越长 → 预测难度越大 → similarity 应下降。
下降越慢 = 模型对物理 dynamics 的理解越好。

============================================================
启动
============================================================
python -m evals.main --fname configs/evals/physion_rollout.yaml --devices cuda:0
============================================================
"""

import os, sys, pickle, logging, csv
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from decord import VideoReader, cpu as decord_cpu

import src.models.vision_transformer as vit
from src.models.predictor import VisionTransformerPredictor
from src.utils.distributed import init_distributed, AllReduce

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED); torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

PHYSION = ['mass', 'friction', 'elasticity', 'deformability']
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Horizon 设置：预测多少 temporal token（总共 8 个 temporal token）
# H=1: 7 context + 1 future  (最简单)
# H=2: 6 context + 2 future
# H=4: 4 context + 4 future  (当前默认)
# H=6: 2 context + 6 future
# H=7: 1 context + 7 future  (最难)
HORIZONS = [1, 2, 3, 4, 5, 6, 7]


def main(args_eval, resume_preempt=False):

    p = args_eval.get('pretrain')
    checkpoint_key = p.get('checkpoint_key', 'target_encoder')
    model_name = p.get('model_name')
    patch_size = p.get('patch_size', 16)
    pretrain_folder = p.get('folder')
    ckp = p.get('checkpoint')
    tag = p.get('write_tag', 'rollout')
    use_sdpa = p.get('use_sdpa', True)
    use_SiLU, tight_SiLU = p.get('use_silu', False), p.get('tight_silu', True)
    uniform_power = p.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp)
    tubelet_size = p.get('tubelet_size', 2)
    pretrain_frames = p.get('frames_per_clip', 16)

    d = args_eval.get('data')
    test_csv = d.get('dataset')
    resolution = args_eval.get('optimization', {}).get('resolution', 224)
    noise_beta = tuple(args_eval.get('noise_beta', [0.5, 1.0]))

    props = args_eval.get('properties', None) or PHYSION
    eval_tag = args_eval.get('tag', 'rollout')
    horizons = args_eval.get('horizons', HORIZONS)

    try: mp.set_start_method('spawn')
    except: pass

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Rank {rank}/{world_size}')

    folder = os.path.join(pretrain_folder, 'physion_rollout/')
    if eval_tag: folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)

    # ---- 加载模型 ----
    ckpt_data = torch.load(pretrained_path, map_location='cpu')
    logger.info(f'Keys: {list(ckpt_data.keys())}')

    encoder = vit.__dict__[model_name](
        img_size=resolution, patch_size=patch_size,
        num_frames=pretrain_frames, tubelet_size=tubelet_size,
        uniform_power=uniform_power, use_sdpa=use_sdpa,
        use_SiLU=use_SiLU, tight_SiLU=tight_SiLU,
    ).to(device).float().eval()
    _load(encoder, ckpt_data, checkpoint_key)
    for p in encoder.parameters(): p.requires_grad = False

    has_pred = 'predictor' in ckpt_data
    predictor = None
    if has_pred:
        ps = _clean(ckpt_data['predictor'])
        pred_dim = ps['predictor_proj.weight'].shape[1]
        pred_depth = sum(1 for k in ps
                        if k.startswith('predictor_blocks') and '.attn.proj.weight' in k)
        predictor = VisionTransformerPredictor(
            img_size=resolution, patch_size=patch_size,
            num_frames=pretrain_frames, tubelet_size=tubelet_size,
            embed_dim=encoder.embed_dim, predictor_embed_dim=pred_dim,
            depth=pred_depth, num_heads=12,
        ).to(device).float().eval()
        predictor.load_state_dict(ps, strict=False)
        for p in predictor.parameters(): p.requires_grad = False
        logger.info(f'Predictor: depth={pred_depth}, dim={pred_dim}')
    else:
        logger.warning('No predictor found — context ablation disabled')

    # ---- 加载视频 ----
    videos = {p: [] for p in props}
    with open(test_csv) as f:
        for row in csv.reader(f):
            if len(row) < 3: continue
            path, prop, label = row[0], row[1].strip(), int(row[2])
            if prop in videos and label >= 0:
                videos[prop].append(path)
    logger.info(f'Videos: {sum(len(v) for v in videos.values())}')

    # ---- 按 token 数 ——
    patch_sz = encoder.patch_embed.patch_size
    patch_sz = patch_sz[0] if isinstance(patch_sz, (tuple, list)) else patch_sz
    grid = resolution // patch_sz
    n_spatial = grid * grid                              # 196
    n_temporal = pretrain_frames // tubelet_size          # 8
    n_total = n_temporal * n_spatial                      # 1568

    logger.info(f'Tokens: {n_temporal} temporal × {n_spatial} spatial = {n_total}')
    logger.info(f'Horizons: {horizons}')

    # ---- 评估（按 horizon） ----
    all_results = {}  # {prop: {horizon: {with_ctx, without_ctx, delta}}}

    for prop in props:
        logger.info(f'\n{"="*60}\nProperty: {prop} ({len(videos[prop])} videos)\n{"="*60}')
        all_results[prop] = _evaluate_property(
            device, encoder, predictor, videos[prop],
            n_spatial, n_temporal, n_total, horizons,
            pretrain_frames, resolution, noise_beta
        )

    # ---- 输出 ----
    if rank == 0:
        _report(folder, tag, all_results, horizons)


# ===========================================================================
# 核心评估
# ===========================================================================

def _evaluate_property(device, encoder, predictor, video_paths,
                       n_spatial, n_temporal, n_total, horizons,
                       num_frames, crop_size, noise_beta):
    """
    对一种物理属性，以所有 horizon 逐视频评估。
    返回: {horizon: {with_ctx_mean, without_ctx_mean, delta_mean, n}}
    """

    np.random.seed(42)
    # 收集数据：{horizon: {'ctx': [tensors], 'fut': [tensors], 'fut_noisy': [tensors]}}
    acc = {h: {'with': [], 'without': [], 'shuffled': []} for h in horizons}
    cache = {h: {'ctx_list': [], 'fut_list': [], 'fut_noisy_list': []} for h in horizons}
    n_valid = 0

    for vi, vpath in enumerate(video_paths):
        sf = _read_sf(vpath)
        if sf is None or sf < num_frames // 2:
            continue

        clip_start = max(0, sf - (num_frames // 2) * 4)
        frames = _load_clip(vpath, clip_start, num_frames, 4, crop_size)
        if frames is None:
            continue

        frames = frames.unsqueeze(0).to(
            device=device, dtype=next(encoder.parameters()).dtype)

        with torch.no_grad():
            all_feat = encoder(frames)

        for h in horizons:
            if h >= n_temporal or h <= 0:
                continue
            ctx_t = n_temporal - h
            ctx_tok = ctx_t * n_spatial

            ctx_idx = torch.arange(ctx_tok, device=device)
            fut_idx = torch.arange(ctx_tok, n_total, device=device)

            ctx_f = all_feat[:, ctx_idx, :]
            fut_f = all_feat[:, fut_idx, :]
            fut_n = predictor.diffusion(fut_f, noise_beta=noise_beta) if predictor else fut_f

            mask_c = [ctx_idx.unsqueeze(0)]
            mask_f = [fut_idx.unsqueeze(0)]

            # ---- 标准评估：先做，同时缓存 ----
            if predictor is not None:
                with torch.no_grad():
                    pred_w = predictor(ctx_f, fut_n, mask_c, mask_f)
                sw = F.cosine_similarity(pred_w[0], fut_f[0], dim=-1).mean().item()

                ctx_z = torch.zeros_like(ctx_f)
                with torch.no_grad():
                    pred_wo = predictor(ctx_z, fut_n, mask_c, mask_f)
                swo = F.cosine_similarity(pred_wo[0], fut_f[0], dim=-1).mean().item()

                acc[h]['with'].append(sw)
                acc[h]['without'].append(swo)

                # 缓存：每 10 个视频采样一次（避免内存爆炸）
                if n_valid % 5 == 0:
                    cache[h]['ctx_list'].append(ctx_f.cpu())
                    cache[h]['fut_list'].append(fut_f.cpu())
                    cache[h]['fut_noisy_list'].append(fut_n.cpu())

        n_valid += 1
        if n_valid % 50 == 0:
            logger.info(f'  [{n_valid}] processed')

    # ---- Shuffled Baseline（打破 context↔future 配对关系） ----
    logger.info('  Computing shuffled baseline...')
    for h in horizons:
        ctx_list = cache[h]['ctx_list']
        fut_list = cache[h]['fut_list']
        fut_n_list = cache[h]['fut_noisy_list']

        if len(ctx_list) < 10:
            continue

        # 随机重排 future，打破正确配对
        indices = np.arange(len(fut_list))
        np.random.shuffle(indices)

        for i in range(len(ctx_list)):
            ctx_f = ctx_list[i].to(device)
            fut_f = fut_list[indices[i]].to(device)
            fut_n = fut_n_list[indices[i]].to(device)

            ctx_tok = ctx_f.shape[1]
            fut_tok = fut_f.shape[1]
            ctx_idx = torch.arange(ctx_tok, device=device)
            fut_idx = torch.arange(ctx_tok, ctx_tok + fut_tok, device=device)
            mask_c = [ctx_idx.unsqueeze(0)]
            mask_f = [fut_idx.unsqueeze(0)]

            with torch.no_grad():
                pred_s = predictor(ctx_f, fut_n, mask_c, mask_f)
            ss = F.cosine_similarity(pred_s[0], fut_f[0], dim=-1).mean().item()
            acc[h]['shuffled'].append(ss)

    # 汇总
    results = {}
    for h in horizons:
        w = acc[h]['with']
        wo = acc[h]['without']
        sh = acc[h]['shuffled']
        if len(w) == 0:
            results[h] = {'with_mean': 0, 'without_mean': 0, 'delta': 0,
                          'shuffled_delta': 0, 'n': 0}
            continue
        delta_real = float(np.mean(w) - np.mean(wo)) if wo else 0
        delta_shuf = float(np.mean(w) - np.mean(sh)) if sh else 0
        results[h] = {
            'with_mean': float(np.mean(w)),
            'without_mean': float(np.mean(wo)) if wo else 0,
            'delta': delta_real,
            'shuffled_mean': float(np.mean(sh)) if sh else 0,
            'shuffled_delta': delta_shuf,
            'n': len(w), 'n_shuf': len(sh),
        }
        _w = results[h]
        logger.info(
            f'  H={h} (ctx={n_temporal-h} tok): '
            f'with={_w["with_mean"]:.4f}  without={_w["without_mean"]:.4f}  '
            f'Δ={_w["delta"]:+.4f}  '
            f'shuf={_w["shuffled_mean"]:.4f}  Δ_shuf={_w["shuffled_delta"]:+.4f}')

    return results


# ===========================================================================
# 输出
# ===========================================================================

def _report(folder, tag, all_results, horizons):
    path = os.path.join(folder, f'{tag}_dynamics_results.txt')
    lines = [
        '=' * 85,
        'V-JEPA Physion++ Latent Dynamics Evaluation',
        '=' * 85,
        '  Δ_real   = WithCtx - WithoutCtx   (context ablation)',
        '  Δ_shuf   = WithCtx - ShuffledPair  (random-pair baseline)',
        '  Δ_real > Δ_shuf 且 Δ_shuf ≈ 0 → Δ 来自 physics, 不是统计假象',
        '',
    ]

    # 按 property 输出
    for prop in PHYSION:
        r = all_results.get(prop, {})
        if not r: continue
        lines.append(f'  --- {prop} ---')
        lines.append(f'  {"H":>5}  {"WithCtx":>8}  {"NoCtx":>8}  '
                     f'{"Δ_real":>8}  {"Shuf":>8}  {"Δ_shuf":>8}  {"N":>5}')
        for h in horizons:
            rh = r.get(h, {})
            if rh.get('n', 0) == 0: continue
            lines.append(
                f'  {h:>5}  {rh["with_mean"]:>8.4f}  '
                f'{rh["without_mean"]:>8.4f}  {rh["delta"]:>+8.4f}  '
                f'{rh["shuffled_mean"]:>8.4f}  {rh["shuffled_delta"]:>+8.4f}  '
                f'{rh["n"]:>5}'
            )
        lines.append('')

    # 汇总：各属性各 horizon 的 Δ_real 和 Δ_shuf
    lines.append('  ' + '=' * 72)
    lines.append('  Cross-property: Δ_real / Δ_shuf')
    lines.append('  (Δ_real should be > Δ_shuf, Δ_shuf should be ≈ 0)')
    header = f'  {"H":>5}'
    for prop in PHYSION:
        header += f'  {prop:>15}'
    lines.append(header)
    for h in horizons:
        row = f'  {h:>5}'
        for prop in PHYSION:
            rh = all_results.get(prop, {}).get(h, {})
            if rh.get('n', 0) > 0:
                row += f'  {rh["delta"]:+.4f}/{rh["shuffled_delta"]:+.4f}'
            else:
                row += f'  {"N/A":>15}'
        lines.append(row)
    lines.append('=' * 85)

    report = '\n'.join(lines)
    print(report)
    with open(path, 'w') as f:
        f.write(report + '\n')
    logger.info(f'Saved: {path}')


# ===========================================================================
# 工具
# ===========================================================================

def _read_sf(vpath):
    try:
        d, n = os.path.dirname(vpath), os.path.basename(vpath)
        pkl = os.path.join(d, f'{n.split("_")[0]}.pkl')
        if not os.path.exists(pkl): return None
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        for k in ['static', 'static0']:
            if k in data and 'start_frame_for_prediction' in data[k]:
                return int(data[k]['start_frame_for_prediction'])
    except: pass
    return None


def _load_clip(vpath, start, n_frames, step, crop_size):
    try:
        vr = VideoReader(vpath, num_threads=1, ctx=decord_cpu(0))
    except: return None
    total = len(vr)
    idx = np.clip(np.arange(start, start + n_frames * step, step),
                  0, total - 1).astype(np.int64)
    try:
        buf = vr.get_batch(idx).asnumpy()
    except: return None

    T, H, W, C = buf.shape
    short = int(crop_size * 256 / 224)
    scl = short / min(H, W)
    nh, nw = int(round(H * scl)), int(round(W * scl))

    resized = np.zeros((T, nh, nw, C), dtype=buf.dtype)
    for t in range(T):
        for i in range(nh):
            si = min(int(i / scl), H - 1)
            resized[t, i] = buf[t, si]
        for j in range(nw):
            sj = min(int(j / scl), W - 1)
            resized[t, :, j] = buf[t, :, sj]

    hs, ws = (nh - crop_size) // 2, (nw - crop_size) // 2
    buf = resized[:, hs:hs + crop_size, ws:ws + crop_size, :]
    buf = buf.astype(np.float32) / 255.0
    buf = (buf - _MEAN) / _STD
    return torch.from_numpy(buf).permute(3, 0, 1, 2)


def _load(model, ckpt, key):
    sd = ckpt.get(key, ckpt.get('encoder', {}))
    sd = _clean(sd)
    msg = model.load_state_dict(sd, strict=False)
    logger.info(f'Loaded {key}: {msg}')


def _clean(sd):
    return {k.replace('module.', '').replace('backbone.', ''): v
            for k, v in sd.items()}
