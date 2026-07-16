#!/usr/bin/env python3
"""
Physion++ 数据准备脚本

从原始 Physion++ 数据（.json + .pkl 格式）生成 V-JEPA 所需的 CSV 索引文件。

【数据下载】
  Physion++ 数据集可以从以下地址下载：
    - train_data:   https://physion-v2.s3.amazonaws.com/train_data.zip
    - readout_data: https://physion-v2.s3.amazonaws.com/readout_data.zip
    - test_data:    https://physion-v2.s3.amazonaws.com/test_data.zip
    - human_data:   https://physion-v2.s3.amazonaws.com/human_data.zip

【数据目录结构】
  下载并解压后，目录结构如下：

  physion_v2/
  ├── train_data/
  │   ├── mass/
  │   │   ├── trial_0000/
  │   │   │   ├── frames/          # 渲染的视频帧 (0000.png, 0001.png, ...)
  │   │   │   ├── trial_0000.json  # 分割掩码
  │   │   │   └── trial_0000.pkl   # 元数据（含 OCP 标签）
  │   │   ├── trial_0001/
  │   │   └── ...
  │   ├── friction/
  │   ├── elasticity/
  │   └── deformability/
  ├── readout_data/      (同上结构)
  └── test_data/         (同上结构)

【使用步骤】

  步骤 1: 渲染视频
    原始 Physion++ 数据只包含仿真状态（JSON+PKL），需要渲染为视频。
    渲染需要使用 ThreeDWorld (TDW) 仿真平台。

    渲染脚本可参考 Physion++ 官方仓库:
      https://github.com/dingmyu/physion_v2

    简单做法：使用官方脚本将每个 trial 渲染为 MP4 视频文件。
    渲染后会得到: trial_0000.mp4, trial_0001.mp4, ...

  步骤 2: 组织目录
    确保目录结构为:
      physion_v2/
      ├── train_data/{mass,friction,elasticity,deformability}/*.mp4
      ├── readout_data/{mass,friction,elasticity,deformability}/*.mp4
      └── test_data/{mass,friction,elasticity,deformability}/*.mp4

  步骤 3: 运行本脚本生成 CSV
    python scripts/prepare_physion_data.py \
        --data_root /path/to/physion_v2 \
        --output_dir /path/to/physion_csv

    这会生成以下 CSV 文件:
      - train_data.csv  (video_path, property_name, label)
      - readout_data.csv
      - test_data.csv
      - train_data_PROPERTY.csv  (每种属性独立的 CSV)
      - readout_data_PROPERTY.csv
      - test_data_PROPERTY.csv

  步骤 4: 在配置文件中填入 CSV 路径
    将生成的 CSV 路径填入 configs/evals/physion_attentive_probe.yaml 的
    dataset_train 和 dataset_val 字段。

【备用方案：无视频时使用帧序列】
  如果没有完整的视频文件，只有帧序列（PNG 图片），可以:
  1. 使用 ffmpeg 将帧序列合成为 MP4:
     ffmpeg -framerate 30 -i frames/%04d.png -c:v libx264 output.mp4
  2. 或者修改数据集类以支持直接读取帧序列（可参考本脚本末尾的注释）

【CSV 格式说明】
  生成的 CSV 每行格式为:
    /absolute/path/to/video.mp4,property_name,label

  其中:
    - property_name: mass | friction | elasticity | deformability
    - label: 0 (NO contact) | 1 (YES contact) | -1 (train_data 无标签)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Physion++ 的 4 种物理属性
PHYSION_PROPERTIES = ['mass', 'friction', 'elasticity', 'deformability']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Physion++ CSV index files for V-JEPA training/evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CSV for all splits
  python prepare_physion_data.py --data_root /data/physion_v2 --output_dir /data/physion_csv

  # Only generate for readout and test splits
  python prepare_physion_data.py --data_root /data/physion_v2 --splits readout test

  # Specify custom video extension
  python prepare_physion_data.py --data_root /data/physion_v2 --video_ext avi
        """,
    )
    parser.add_argument(
        '--data_root', type=str, required=True,
        help='Physion++ 数据根目录（包含 train_data/, readout_data/, test_data/）'
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
        '--properties', nargs='+', default=PHYSION_PROPERTIES,
        help='要处理的物理属性 (默认: 全部 4 种)'
    )
    parser.add_argument(
        '--video_ext', type=str, default='mp4',
        help='视频文件扩展名 (默认: mp4)'
    )
    parser.add_argument(
        '--use_pkl_labels', action='store_true', default=True,
        help='从 .pkl 元数据文件读取标签（默认开启）'
    )
    parser.add_argument(
        '--no_labels', action='store_true',
        help='不读取标签（所有标签设为 -1，适用于无标签的预训练数据）'
    )
    return parser.parse_args()


def extract_label_from_pkl(pkl_path):
    """
    从 Physion++ 的 .pkl 元数据文件中提取 OCP 标签。

    .pkl 文件包含:
      - friction_coefficient / mass / elasticity / deformability
      - object_positions, object_rotations, object_velocities
      - camera_matrix
      - collision_events (bool)
      - trial_seed
      - label: 0 (NO contact) 或 1 (YES contact)
      - start_frame_for_prediction

    返回:
        int: 0 (NO), 1 (YES), 或 -1 (未找到)
    """
    try:
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)

        # 尝试多种可能的 label 键名
        for key in ['label', 'ocp_label', 'contact_label', 'outcome']:
            if key in metadata:
                return int(metadata[key])

        # 如果元数据中没有显式的 label 字段，
        # 可以尝试从 collision_events 推断
        if 'collision' in metadata:
            collision = metadata['collision']
            if isinstance(collision, bool):
                return 1 if collision else 0
            elif isinstance(collision, (list, tuple)):
                return 1 if any(collision) else 0

        print(f'Warning: No label found in {pkl_path}, returning -1')
        return -1

    except Exception as e:
        print(f'Warning: Failed to read {pkl_path}: {e}')
        return -1


def find_videos_in_directory(trial_dir, video_ext='mp4'):
    """
    在 trial 目录中查找视频文件。

    查找策略（按优先级）:
      1. trial_XXXX.mp4 (直接渲染的视频文件)
      2. frames/ 目录下的帧序列 (返回 None，需要另外处理)
      3. 任何 .mp4/.avi 文件

    返回:
        str: 视频文件路径，或 None（未找到）
    """
    # 策略 1: 在 trial 目录直接查找 mp4 文件
    video_files = list(trial_dir.glob(f'*.{video_ext}'))
    if video_files:
        return str(video_files[0])

    # 策略 2: 检查常见的视频文件名模式
    for pattern in [f'*.{video_ext}', f'*.{video_ext.upper()}']:
        video_files = list(trial_dir.glob(pattern))
        if video_files:
            return str(video_files[0])

    return None


def generate_csv_for_split(data_root, split_name, output_dir, properties, video_ext, no_labels):
    """
    为一个数据划分生成 CSV 索引文件。

    参数:
        data_root: Physion++ 数据根目录
        split_name: 'train', 'readout', 'test'
        output_dir: CSV 输出目录
        properties: 物理属性列表
        video_ext: 视频扩展名
        no_labels: True 时所有标签设为 -1

    返回:
        dict: {property_name: [(video_path, label), ...]}
    """

    split_dir = Path(data_root) / f'{split_name}_data'
    if not split_dir.exists():
        print(f'Warning: Directory not found: {split_dir}')
        return {}

    all_entries = []
    prop_entries = {p: [] for p in properties}
    skipped_no_video = 0
    skipped_no_label = 0

    for prop in properties:
        prop_dir = split_dir / prop
        if not prop_dir.exists():
            print(f'Warning: Property directory not found: {prop_dir}')
            continue

        # 查找所有 trial 目录或视频文件
        # 模式 1: trial_XXXX/ 子目录 (原始结构)
        trial_dirs = sorted([
            d for d in prop_dir.iterdir()
            if d.is_dir() and d.name.startswith('trial_')
        ])

        # 模式 2: 直接在属性目录下的视频文件 (简化结构)
        direct_videos = sorted([
            f for f in prop_dir.iterdir()
            if f.is_file() and f.suffix.lstrip('.') in [video_ext, video_ext.lower(), video_ext.upper()]
        ])

        if trial_dirs:
            # 原始结构: 每个 trial 一个子目录
            for trial_dir in trial_dirs:
                video_path = find_videos_in_directory(trial_dir, video_ext)

                if video_path is None:
                    skipped_no_video += 1
                    print(f'  No video found in: {trial_dir}')
                    continue

                # 从 .pkl 文件中提取标签
                label = -1
                if not no_labels:
                    pkl_files = list(trial_dir.glob('*.pkl'))
                    if pkl_files:
                        label = extract_label_from_pkl(str(pkl_files[0]))
                    else:
                        skipped_no_label += 1

                all_entries.append((video_path, prop, label))
                prop_entries[prop].append((video_path, prop, label))

        elif direct_videos:
            # 简化结构: 视频直接放在属性目录下
            for video_file in direct_videos:
                video_path = str(video_file)

                # 尝试从同目录的 .pkl 文件获取标签
                label = -1
                if not no_labels:
                    trial_name = video_file.stem
                    pkl_path = Path(prop_dir) / f'{trial_name}.pkl'
                    if pkl_path.exists():
                        label = extract_label_from_pkl(str(pkl_path))
                    else:
                        skipped_no_label += 1

                all_entries.append((video_path, prop, label))
                prop_entries[prop].append((video_path, prop, label))

        else:
            print(f'Warning: No trials found in: {prop_dir}')

    # ---- 写入 CSV 文件 ----

    os.makedirs(output_dir, exist_ok=True)

    def write_csv(filename, entries):
        """写入 CSV 文件"""
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for video_path, prop, label in entries:
                f.write(f'{video_path},{prop},{label}\n')
        print(f'  Created: {filepath} ({len(entries)} entries)')
        return filepath

    # 全部属性合并的 CSV
    if all_entries:
        write_csv(f'{split_name}_data.csv', all_entries)

    # 每种属性独立的 CSV
    for prop in properties:
        if prop_entries[prop]:
            write_csv(f'{split_name}_data_{prop}.csv', prop_entries[prop])

    # 统计
    if skipped_no_video > 0:
        print(f'  Warning: {skipped_no_video} trials skipped (no video found)')
    if skipped_no_label > 0:
        print(f'  Warning: {skipped_no_label} trials have missing labels')

    # 打印统计信息
    for prop in properties:
        entries = prop_entries[prop]
        if entries:
            labels = [e[2] for e in entries if e[2] >= 0]
            if labels:
                yes_count = sum(labels)
                no_count = len(labels) - yes_count
                print(f'  [{prop}] {len(entries)} trials: YES={yes_count}, NO={no_count}')
            else:
                print(f'  [{prop}] {len(entries)} trials (unlabeled)')

    return prop_entries


def main():
    args = parse_args()

    print('=' * 60)
    print('Physion++ Data Preparation for V-JEPA')
    print('=' * 60)
    print(f'Data root:  {args.data_root}')
    print(f'Output dir: {args.output_dir}')
    print(f'Splits:     {args.splits}')
    print(f'Properties: {args.properties}')
    print(f'Video ext:  {args.video_ext}')
    print(f'Use labels: {not args.no_labels}')
    print('=' * 60)
    print()

    for split in args.splits:
        print(f'\n--- Processing: {split}_data ---')
        generate_csv_for_split(
            data_root=args.data_root,
            split_name=split,
            output_dir=args.output_dir,
            properties=args.properties,
            video_ext=args.video_ext,
            no_labels=args.no_labels,
        )

    print('\n' + '=' * 60)
    print('Done! CSV files generated in:', args.output_dir)
    print()
    print('Next steps:')
    print('  1. Update configs/evals/physion_attentive_probe.yaml:')
    print(f'     dataset_train: {args.output_dir}/readout_data.csv')
    print(f'     dataset_val:   {args.output_dir}/test_data.csv')
    print('  2. Run the evaluation:')
    print('     python -m evals.main --fname configs/evals/physion_attentive_probe.yaml --devices cuda:0')
    print('=' * 60)


if __name__ == '__main__':
    main()
