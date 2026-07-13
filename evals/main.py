# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 评估入口（本地多GPU）
# ============================================================================
# 这个文件是 V-JEPA 评估的本地启动入口。
# 与预训练入口(app/main.py)结构类似，但用于启动评估任务。
#
# 评估类型由配置文件中的 eval_name 字段决定：
#   - 'image_classification_frozen' → 图像分类（冻结backbone）
#   - 'video_classification_frozen' → 视频分类（冻结backbone）
#
# 使用方式：
#   python -m evals.main --fname configs/evals/vith16_in1k.yaml --devices cuda:0 cuda:1 cuda:2

import argparse
import multiprocessing as mp
import pprint
import yaml

from src.utils.distributed import init_distributed
from evals.scaffold import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='评估配置文件路径（YAML格式）',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='本地使用的GPU设备列表')


def process_main(rank, fname, world_size, devices):
    """
    每个GPU进程的主函数

    流程：
    1. 设置可见GPU
    2. 加载YAML配置文件
    3. 初始化分布式通信
    4. 启动评估任务
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # 加载配置文件
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # 初始化分布式通信
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # 启动评估
    # eval_name 决定使用哪种评估任务（如 'image_classification_frozen'）
    eval_main(params['eval_name'], args_eval=params)


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()
