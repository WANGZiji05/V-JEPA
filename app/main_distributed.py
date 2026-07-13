# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 分布式训练入口（SLURM 集群）
# ============================================================================
# 这个文件是 V-JEPA 预训练的分布式启动入口。
# 使用 Facebook 的 submitit 库在 SLURM 集群上启动大规模分布式训练。
#
# 与 app/main.py 的区别：
# - main.py: 单机多GPU，用 multiprocessing 启动进程
# - main_distributed.py: 多机多GPU，用 submitit 向 SLURM 提交作业
#
# 使用方式：
#   python -m app.main_distributed \
#     --fname configs/pretrain/vitl16.yaml \
#     --folder /path/to/logs \
#     --partition $SLURM_PARTITION
#
# submitit 的作用：
# 1. 自动向 SLURM 提交作业
# 2. 管理作业的生命周期（提交、监控、重试）
# 3. 支持抢占式调度（preemption）：作业被中断后自动恢复

import argparse
import os
import pprint
import yaml

import submitit  # Facebook的集群作业管理库

from app.scaffold import main as app_main
from src.utils.logging import get_logger

logger = get_logger(force=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='submitit日志保存目录',
    default='/fsx-jepa/massran/submitit/')
parser.add_argument(
    '--exclude', type=str,
    help='要排除的计算节点（逗号分隔）',
    default=None)
parser.add_argument(
    '--batch-launch', action='store_true',
    help='--fname是否指向一个包含多个配置文件列表的yaml文件（批量启动）')
parser.add_argument(
    '--fname', type=str,
    help='配置文件路径（YAML格式）',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='SLURM分区名称')
parser.add_argument(
    '--time', type=int, default=4300,
    help='作业运行时间（分钟），默认4300分钟=~3天')


class Trainer:
    """
    训练器类 —— 封装一次训练作业

    submitit 要求提交的作业是一个可调用对象。
    Trainer将配置和训练逻辑包装成submitit兼容的格式。

    关键方法：
    - __call__(): submitit在执行作业时调用此方法
    - checkpoint(): 作业被抢占时，submitit调用此方法保存状态
    """

    def __init__(self, args_pretrain, load_model=None):
        self.app = args_pretrain['app']      # 任务类型（如'vjepa'）
        self.args_pretrain = args_pretrain   # 配置参数
        self.load_model = load_model          # 是否加载checkpoint

    def __call__(self):
        """submitit 执行入口 —— 启动训练"""
        app = self.app
        params = self.args_pretrain
        load_model = self.load_model

        logger.info('loaded pretrain params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

        # 启动训练
        resume_preempt = False if load_model is None else load_model
        app_main(app, args=params, resume_preempt=resume_preempt)

    def checkpoint(self):
        """
        作业被抢占时的checkpoint保存

        当SLURM因为时间限制或优先级原因终止作业时，
        submitit会调用此方法。
        返回的DelayedSubmission允许作业自动重新排队和恢复。
        """
        fb_trainer = Trainer(self.args_pretrain, True)  # True=从checkpoint恢复
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_app_with_parsed_args(
    args_for_pretrain,   # 配置参数列表
    submitit_folder,     # submitit日志目录
    partition,           # SLURM分区
    timeout=4300,        # 超时时间（分钟）
    nodes=1,             # 节点数
    tasks_per_node=1,    # 每节点GPU数
    exclude_nodes=None   # 排除的节点
):
    """
    使用 submitit 向 SLURM 提交训练作业

    submitit.AutoExecutor:
    - 自动创建SLURM作业脚本
    - 管理作业提交、监控和重试
    - 支持作业数组（一次提交多个作业）
    """
    executor = submitit.AutoExecutor(
        folder=os.path.join(submitit_folder, 'job_%j'),  # %j=作业ID
        slurm_max_num_timeout=20)  # 最多重试20次

    # 设置SLURM作业参数
    executor.update_parameters(
        slurm_partition=partition,        # SLURM分区
        slurm_mem_per_gpu='55G',          # 每个GPU的内存
        timeout_min=timeout,              # 作业超时时间
        nodes=nodes,                      # 计算节点数
        tasks_per_node=tasks_per_node,    # 每节点任务数（=GPU数）
        cpus_per_task=12,                 # 每个任务的CPU数
        gpus_per_node=tasks_per_node)     # 每节点GPU数

    if args.exclude is not None:
        executor.update_parameters(slurm_exclude=args.exclude)

    # 批量提交作业
    jobs, trainers = [], []
    with executor.batch():  # 批量提交，共享作业数组ID
        for ap in args_for_pretrain:
            fb_trainer = Trainer(ap)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)  # 打印SLURM作业ID


def launch():
    """
    启动分布式训练的主函数

    流程：
    1. 读取配置文件列表
    2. 解析每个YAML配置文件
    3. 提取节点数和GPU数
    4. 调用submitit提交作业
    """

    # 步骤1：准备配置文件列表
    config_fnames = [args.fname]

    # 如果 --batch-launch 为 True，fname指向的文件包含多个配置文件的列表
    if args.batch_launch:
        with open(args.fname, 'r') as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)

    # 步骤2：解析每个配置文件
    nodes, tasks_per_node = None, None
    configs = []
    for f in config_fnames:
        with open(f, 'r') as y_file:
            _params = yaml.load(y_file, Loader=yaml.FullLoader)
            nodes = int(_params.get('nodes'))
            tasks_per_node = int(_params.get('tasks_per_node'))
            configs += [_params]
    logger.info(f'Loaded {len(configs)} config files')
    logger.info(f'Running all jobs with {nodes=} / {tasks_per_node=}')

    # 步骤3：提交作业
    launch_app_with_parsed_args(
        args_for_pretrain=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude)


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
