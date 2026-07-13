# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 本地训练入口（单机多GPU）
# ============================================================================
# 这个文件是 V-JEPA 预训练的本地启动入口。
# 当你在一台机器上用多个GPU训练时使用此入口。
#
# 使用方式：
#   python -m app.main --fname configs/pretrain/vitl16.yaml --devices cuda:0 cuda:1 cuda:2
#
# 它会：
# 1. 读取YAML配置文件
# 2. 为每个GPU启动一个独立的进程
# 3. 每个进程运行 app/vjepa/train.py 中的训练循环
# 4. 进程间通过 PyTorch Distributed 进行通信（梯度同步等）

import argparse
import multiprocessing as mp
import pprint
import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='配置文件路径（YAML格式）',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='本地使用的GPU设备列表，如 cuda:0 cuda:1 cuda:2')


def process_main(rank, fname, world_size, devices):
    """
    每个GPU进程的主函数

    参数:
        rank: 当前进程的排名（0, 1, 2, ...）
        fname: 配置文件路径
        world_size: 总GPU数量
        devices: GPU设备列表
    """
    import os
    # 设置当前进程可见的GPU（每个进程只能看到分配给它的那张GPU）
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)
    # 只有主进程（rank=0）打印INFO级别日志，其他进程只打印ERROR
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # 加载YAML配置文件
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # 主进程打印配置并保存副本
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    # 初始化分布式通信（同一台机器上多个GPU之间的通信）
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # 启动训练！调用 app/vjepa/train.py 中的 main 函数
    app_main(params['app'], args=params)


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)

    # 使用spawn方式启动多进程（避免CUDA初始化问题）
    mp.set_start_method('spawn')

    # 为每个GPU启动一个独立的训练进程
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()
