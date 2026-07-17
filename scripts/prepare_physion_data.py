#!/usr/bin/env python3
"""
Physion++ 数据准备脚本（适配实际目录结构 v2.0）

从 Physion++ 的实际目录结构生成 V-JEPA 所需的 CSV 索引文件。

【实际目录结构】
  data_root/
  ├── data_v1/                        # 训练集
  │   ├── bouncy_wall_pp/             # scenario 文件夹
  │   │   └── bouncy_wall-zld=0-target=dumbbell/  # config 子文件夹
  │   │       ├── 0000_img.mp4        # RGB 视频 ← 我们用这个
  │   │       ├── 0000_id.mp4         # 分割掩码视频（忽略）
  │   │       ├── 0000_id.json        # 分割数据（忽略）
  │   │       ├── 0000_map.png        # 可视化图（忽略）
  │   │       └── 0000.pkl            # 元数据（含 OCP 标签）
  │   ├── deform_clothhit_pp/
  │   ├── friction_platform_pp/
  │   ├── mass_waterpush_pp/
  │   └── ... (更多 scenario 文件夹)
  │
  ├── readout_data_v1/                # 读出色合（同上结构）
  │   └── ... (同 data_v1 的场景)
  │
  └── testdata_v1/                    # 测试集
      ├── bouncy_platform_pp-copy0/   # 配对试验 - 副本 0
      ├── bouncy_platform_pp-copy1/   # 配对试验 - 副本 1
      ├── ... (每场景都有 -copy0 和 -copy1)
      ├── physionpp-bouncy_merge_230108.csv     # bouncy(elasticity) 标签
      ├── physionpp-deform_merge_230120.csv     # deform(deformability) 标签
      ├── physionpp-friction_merge_230108.csv   # friction 标签
      └── physionpp-mass_merge_221111.csv       # mass 标签

【测试集 CSV 格式 (merge CSV)】
  full_stim_paths, full_stim_apaths, filenames, filenames_a,
  target_hit_zone_labels, start_frame_for_prediction

  其中 target_hit_zone_labels (True/False) 就是 OCP 标签。

【物理属性映射 (scenario 前缀 → 属性名)】
  bouncy_*  → elasticity    (弹性/反弹)
  deform_*  → deformability (可变形性)
  friction_* → friction     (摩擦力)
  mass_*    → mass         (质量)

【使用方法】
  python scripts/prepare_physion_data.py \
      --data_root /path/to/your/data/ \
      --train_dir data_v1 \
      --readout_dir readout_data_v1 \
      --test_dir testdata_v1 \
      --output_dir ./physion_csv
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path


# ─── 物理属性映射：scenario 前缀 → 属性名 ───
PREFIX_TO_PROPERTY = {
    'bouncy': 'elasticity',
    'deform': 'deformability',
    'friction': 'friction',
    'mass': 'mass',
}

PHYSION_PROPERTIES = ['mass', 'friction', 'elasticity', 'deformability']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Physion++ CSV index files for V-JEPA (v2 actual structure)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 处理全部三个 split
  python scripts/prepare_physion_data.py \\
      --data_root D:/datasets/physion_v2 \\
      --output_dir ./physion_csv

  # 只处理训练集（用于自监督预训练）
  python scripts/prepare_physion_data.py \\
      --data_root D:/datasets/physion_v2 \\
      --splits train \\
      --output_dir ./physion_csv

  # 只处理 readout + test（用于探针评估）
  python scripts/prepare_physion_data.py \\
      --data_root D:/datasets/physion_v2 \\
      --splits readout test \\
      --output_dir ./physion_csv
        """,
    )
    parser.add_argument(
        '--data_root', type=str, required=True,
        help='Physion++ 数据根目录（包含 data_v1/, readout_data_v1/, testdata_v1/）'
    )
    parser.add_argument(
        '--train_dir', type=str, default='data_v1',
        help='训练数据子目录名 (默认: data_v1)'
    )
    parser.add_argument(
        '--readout_dir', type=str, default='readout_data_v1',
        help='读出数据子目录名 (默认: readout_data_v1)'
    )
    parser.add_argument(
        '--test_dir', type=str, default='testdata_v1',
        help='测试数据子目录名 (默认: testdata_v1)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./physion_csv',
        help='CSV 文件输出目录 (默认: ./physion_csv)'
    )
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'readout', 'test'],
        help='要处理的数据划分 (默认: train readout test)'
    )
    parser.add_argument(
        '--no_labels', action='store_true',
        help='不读取标签（训练集无监督预训练时可用，所有 label 设为 -1）'
    )
    parser.add_argument(
        '--videodataset_format', action='store_true',
        help='额外输出 VideoDataset 兼容格式（空格分隔、2列: video_path label）'
        ' 用于标准 V-JEPA 预训练管线'
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def scenario_to_property(scenario_name):
    """
    根据 scenario 名称的前缀推断物理属性类别。

    例如:
      'bouncy_wall_pp'       → 'elasticity'
      'friction_collision_pp' → 'friction'
      'mass_dominoes_pp'      → 'mass'
      'deform_clothhit_pp'    → 'deformability'
    """
    for prefix, prop in PREFIX_TO_PROPERTY.items():
        if scenario_name.startswith(prefix):
            return prop
    return 'unknown'


def extract_label_from_pkl(pkl_path):
    """
    从 .pkl 元数据文件中提取 OCP 标签。

    Physion++ 实际数据结构:
      data['static']['does_target_contact_zone']  → bool (OCP label)
      data['static0']['does_target_contact_zone']  → 部分 readout 数据用此键
      data['frames'][last_frame]['labels']['target_contacting_zone'] → 逐帧标签

    返回:
        int: 0 (NO contact), 1 (YES contact), 或 -1 (未找到)
    """
    try:
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)

        # ── 方法 1: static 中的全局标签 ──
        for static_key in ['static', 'static0']:
            if static_key in metadata:
                static = metadata[static_key]
                if 'does_target_contact_zone' in static:
                    val = static['does_target_contact_zone']
                    if isinstance(val, bool):
                        return 1 if val else 0
                    return int(val)

        # ── 方法 2: 从最后一帧的 labels 推断 ──
        if 'frames' in metadata:
            frames = metadata['frames']
            if frames:
                last_frame_key = sorted(frames.keys(), key=int)[-1]
                last_frame = frames[last_frame_key]
                if 'labels' in last_frame:
                    lbl = last_frame['labels']
                    if 'target_contacting_zone' in lbl:
                        val = lbl['target_contacting_zone']
                        if isinstance(val, bool):
                            return 1 if val else 0
                        return int(val)

        return -1  # 未找到标签

    except Exception as e:
        print(f'  [WARN] Failed to read {pkl_path}: {e}')
        return -1


def strip_copy_suffix(scenario_name):
    """
    去除测试集 scenario 目录名的 -copy0/-copy1 后缀。

    例如:
      'bouncy_platform_pp-copy0' → 'bouncy_platform_pp'
      'mass_dominoes_pp-copy1'   → 'mass_dominoes_pp'
    """
    return re.sub(r'-copy\d+$', '', scenario_name)


# ═══════════════════════════════════════════════════════════════════════════
# 各 split 的扫描逻辑
# ═══════════════════════════════════════════════════════════════════════════

def scan_train_or_readout(split_dir, no_labels):
    """
    扫描训练集或读出集。

    目录结构: split_dir/{scenario}/{config_subfolder}/*_img.mp4

    对每个 _img.mp4 视频:
      - 同目录下找同编号的 .pkl 文件提取标签
      - 根据 scenario 名推断物理属性

    返回:
        all_entries: [(video_abs_path, property_name, label), ...]
        prop_entries: {property_name: [(video_abs_path, label), ...]}
    """
    all_entries = []
    prop_entries = {p: [] for p in PHYSION_PROPERTIES}
    skipped = {'no_video': 0, 'no_label': 0}

    split_path = Path(split_dir)
    if not split_path.exists():
        print(f'  [WARN] Directory not found: {split_path}')
        return all_entries, prop_entries, skipped

    # 遍历 scenario 文件夹
    scenario_dirs = sorted(
        [d for d in split_path.iterdir() if d.is_dir()]
    )

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        prop = scenario_to_property(scenario_name)

        # 遍历 config 子文件夹
        config_dirs = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir()]
        )

        for config_dir in config_dirs:
            # 找所有 _img.mp4 (RGB 视频，跳过 _id.mp4)
            video_files = sorted(config_dir.glob('*_img.mp4'))

            if not video_files:
                skipped['no_video'] += 1
                continue

            for video_file in video_files:
                video_path = str(video_file.resolve())

                # 找对应的 .pkl 文件
                # video 文件名: 0000_img.mp4 → pkl: 0000.pkl
                trial_num = video_file.name.split('_')[0]  # e.g., "0000"
                pkl_path = config_dir / f'{trial_num}.pkl'

                label = -1
                if not no_labels:
                    if pkl_path.exists():
                        label = extract_label_from_pkl(str(pkl_path))
                    else:
                        skipped['no_label'] += 1

                all_entries.append((video_path, prop, label))
                if prop in prop_entries:
                    prop_entries[prop].append((video_path, prop, label))

    return all_entries, prop_entries, skipped


def scan_test(split_dir, no_labels):
    """
    扫描测试集。

    测试集特殊之处:
      1. 每个 scenario 有 -copy0 和 -copy1 配对目录
      2. 标签在顶层 merge CSV 文件中，不在 .pkl 中
      3. merge CSV 以物理属性命名: physionpp-{property}_merge_*.csv

    merge CSV 格式:
      full_stim_paths, full_stim_apaths, filenames, filenames_a,
      target_hit_zone_labels, start_frame_for_prediction

    返回:
        all_entries, prop_entries, skipped
    """
    import csv as csv_module

    all_entries = []
    prop_entries = {p: [] for p in PHYSION_PROPERTIES}
    skipped = {'no_video': 0, 'no_label': 0, 'no_csv_match': 0}

    split_path = Path(split_dir)
    if not split_path.exists():
        print(f'  [WARN] Directory not found: {split_path}')
        return all_entries, prop_entries, skipped

    # ── 步骤 1: 读取所有 merge CSV 文件 ──
    # filename → (property, label) 映射
    filename_label_map = {}

    csv_files = sorted(split_path.glob('physionpp-*_merge_*.csv'))
    if not csv_files:
        print('  [WARN] No merge CSV files found in test directory!')
        print('  Falling back to PKL-based label extraction...')
        return scan_train_or_readout(split_dir, no_labels)

    for csv_path in csv_files:
        csv_name = csv_path.name.lower()

        # 从 CSV 文件名推断物理属性
        prop = 'unknown'
        for prefix, p in PREFIX_TO_PROPERTY.items():
            if prefix in csv_name:
                prop = p
                break

        print(f'  Reading merge CSV: {csv_path.name} → property: {prop}')

        with open(csv_path, 'r') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                # filenames 列: e.g.,
                #   "bouncy_platform_pp-copy0_bouncy_platform-...-0000_img.mp4"
                filename = row.get('filenames', '').strip()
                label_str = row.get('target_hit_zone_labels', '').strip()

                if not filename:
                    continue

                # 解析标签
                if no_labels:
                    label = -1
                elif label_str.lower() in ('true', '1', 'yes'):
                    label = 1
                elif label_str.lower() in ('false', '0', 'no'):
                    label = 0
                else:
                    label = -1
                    skipped['no_label'] += 1

                filename_label_map[filename] = (prop, label)

    print(f'  Loaded {len(filename_label_map)} entries from merge CSVs')

    # ── 步骤 2: 遍历目录找到实际的视频文件并匹配标签 ──
    scenario_dirs = sorted(
        [d for d in split_path.iterdir()
         if d.is_dir() and 'copy' in d.name.lower()]
    )

    if not scenario_dirs:
        # 可能子目录就是 config 目录，不是 scenario 目录
        # 尝试 scan_train_or_readout 的目录结构
        print('  No -copy directories found, trying flat scan...')
        return scan_train_or_readout(split_dir, no_labels)

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        base_scenario = strip_copy_suffix(scenario_name)
        prop = scenario_to_property(base_scenario)

        # 遍历 config 子文件夹
        config_dirs = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir()]
        )

        for config_dir in config_dirs:
            video_files = sorted(config_dir.glob('*_img.mp4'))

            if not video_files:
                skipped['no_video'] += 1
                continue

            for video_file in video_files:
                video_path = str(video_file.resolve())

                # 在 merge CSV 的 filename 列中匹配
                # merge CSV 中的 filenames 格式:
                #   {scenario-copy}_{config}_{trial_num}_img.mp4
                # 我们从路径中构造同样的格式来匹配
                label = -1

                if not no_labels:
                    # 构造 merge CSV 中的 filenames 格式
                    # e.g., "bouncy_platform_pp-copy0_bouncy_platform-..._0000_img.mp4"
                    trial_name = video_file.name  # "0000_img.mp4"
                    expected_filename = (
                        f'{scenario_name}_'
                        f'{config_dir.name}_'
                        f'{trial_name}'
                    )

                    # 尝试匹配
                    if expected_filename in filename_label_map:
                        matched_prop, label = filename_label_map[expected_filename]
                    else:
                        # 模糊匹配：忽略属性名差异
                        matched = False
                        for fname, (fprop, flabel) in filename_label_map.items():
                            if trial_name in fname and config_dir.name in fname:
                                label = flabel
                                matched = True
                                break
                        if not matched:
                            skipped['no_csv_match'] += 1

                all_entries.append((video_path, prop, label))
                if prop in prop_entries:
                    prop_entries[prop].append((video_path, prop, label))

    return all_entries, prop_entries, skipped


# ═══════════════════════════════════════════════════════════════════════════
# CSV 写入
# ═══════════════════════════════════════════════════════════════════════════

def write_csv_files(output_dir, split_name, all_entries, prop_entries):
    """将扫描结果写入 CSV 文件。"""

    os.makedirs(output_dir, exist_ok=True)

    def _write(filename, entries):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for video_path, prop, label in entries:
                f.write(f'{video_path},{prop},{label}\n')
        print(f'  Created: {filepath} ({len(entries)} entries)')
        return filepath

    # 全部属性合并的 CSV
    if all_entries:
        _write(f'{split_name}_data.csv', all_entries)

    # 每种属性独立的 CSV
    for prop in PHYSION_PROPERTIES:
        if prop_entries.get(prop):
            _write(f'{split_name}_data_{prop}.csv', prop_entries[prop])


def write_videodataset_format(output_dir, split_name, all_entries, prop_entries):
    """
    额外输出 VideoDataset 兼容格式的 CSV。

    标准 VideoDataset 期望格式：空格分隔、2 列
      /path/to/video.mp4 integer_label

    预训练时 label 会被忽略（设为 0），评估时需要真实 label。
    """

    os.makedirs(output_dir, exist_ok=True)

    def _write_vd(filename, entries):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for video_path, prop, label in entries:
                # VideoDataset 格式: 空格分隔
                # label 取 max(0, label) 确保 -1 变为 0
                safe_label = label if label >= 0 else 0
                f.write(f'{video_path} {safe_label}\n')
        print(f'  Created (VideoDataset fmt): {filepath} ({len(entries)} entries)')
        return filepath

    # 全部属性合并
    if all_entries:
        _write_vd(f'{split_name}_data_vd.csv', all_entries)

    # 每种属性独立
    for prop in PHYSION_PROPERTIES:
        if prop_entries.get(prop):
            _write_vd(f'{split_name}_data_{prop}_vd.csv', prop_entries[prop])


def print_statistics(prop_entries, skipped):
    """打印扫描统计信息。"""
    total = 0
    for prop in PHYSION_PROPERTIES:
        entries = prop_entries.get(prop, [])
        if not entries:
            continue
        total += len(entries)
        labels = [e[2] for e in entries if e[2] >= 0]
        if labels:
            yes_c = sum(labels)
            no_c = len(labels) - yes_c
            print(f'  [{prop:15s}] {len(entries):5d} trials  '
                  f'YES={yes_c:5d}  NO={no_c:5d}')
        else:
            print(f'  [{prop:15s}] {len(entries):5d} trials  (unlabeled)')

    print(f'  {"─" * 50}')
    print(f'  Total: {total} trials')

    if skipped.get('no_video', 0) > 0:
        print(f'  [WARN] {skipped["no_video"]} config dirs: no _img.mp4 found')
    if skipped.get('no_label', 0) > 0:
        print(f'  [WARN] {skipped["no_label"]} trials: missing .pkl label')
    if skipped.get('no_csv_match', 0) > 0:
        print(f'  [WARN] {skipped["no_csv_match"]} trials: '
              f'video not matched in merge CSV')


# ═══════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Split 名称 → (目录名, 扫描函数)
    SPLIT_CONFIG = {
        'train':   (args.train_dir,   scan_train_or_readout),
        'readout': (args.readout_dir, scan_train_or_readout),
        'test':    (args.test_dir,    scan_test),
    }

    print('=' * 60)
    print('Physion++ Data Preparation for V-JEPA (v2 structure)')
    print('=' * 60)
    print(f'Data root:    {args.data_root}')
    print(f'Train dir:    {args.train_dir}')
    print(f'Readout dir:  {args.readout_dir}')
    print(f'Test dir:     {args.test_dir}')
    print(f'Output dir:   {args.output_dir}')
    print(f'Splits:       {args.splits}')
    print(f'Use labels:   {not args.no_labels}')
    print(f'VD format:    {args.videodataset_format}')
    print('=' * 60)

    for split in args.splits:
        if split not in SPLIT_CONFIG:
            print(f'\n[WARN] Unknown split "{split}", skipping')
            continue

        dir_name, scan_fn = SPLIT_CONFIG[split]
        split_path = os.path.join(args.data_root, dir_name)

        print(f'\n{"─" * 60}')
        print(f'Processing: {split} ({dir_name})')
        print(f'Path: {split_path}')
        print(f'{"─" * 60}')

        all_entries, prop_entries, skipped = scan_fn(split_path, args.no_labels)

        write_csv_files(args.output_dir, split, all_entries, prop_entries)
        if args.videodataset_format:
            write_videodataset_format(
                args.output_dir, split, all_entries, prop_entries)
        print_statistics(prop_entries, skipped)

    # ── 最终指引 ──
    print('\n' + '=' * 60)
    print('Done! CSV files generated in:', os.path.abspath(args.output_dir))
    print()
    print('Next steps:')
    print('  1. Edit configs/evals/physion_attentive_probe.yaml:')
    print(f'     dataset_train: {os.path.abspath(args.output_dir)}/readout_data.csv')
    print(f'     dataset_val:   {os.path.abspath(args.output_dir)}/test_data.csv')
    print()
    if args.videodataset_format:
        print('  2. For pretraining (use VideoDataset-compatible files):')
        print(f'     configs/pretrain/physion_vith16.yaml →')
        print(f'     data.datasets: [{os.path.abspath(args.output_dir)}/train_data_vd.csv]')
        print()
        print('  3. Run the attentive probe evaluation:')
    else:
        print('  2. Run the attentive probe evaluation:')
    print('     python -m evals.main \\')
    print('         --fname configs/evals/physion_attentive_probe.yaml \\')
    print('         --devices cuda:0')
    print('=' * 60)


if __name__ == '__main__':
    main()
