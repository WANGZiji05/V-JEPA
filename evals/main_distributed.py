# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# V-JEPA 分布式评估入口（SLURM 集群）
# ============================================================================
# 与 app/main_distributed.py 类似，但用于评估任务。
# 使用 submitit 向 SLURM 集群提交评估作业。
#
# 评估比预训练需要的资源更少（编码器被冻结），
# 但大规模视频数据集仍然需要分布式处理来加速。
#
# 使用方式：
#   python -m evals.main_distributed \
#     --fname configs/evals/vith16_in1k.yaml \
#     --folder /path/to/logs \
#     --partition $SLURM_PARTITION

import argparse
import logging
import os
import pprint
import sys
import time
import yaml

import submitit  # Facebook的集群作业管理库

from evals.scaffold import main as eval_main

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

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
    help='--fname是否指向一个包含多个配置文件列表的yaml文件（批量启动多个评估）')
parser.add_argument(
    '--fname', type=str,
    help='配置文件路径（YAML格式）',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='SLURM分区名称')
parser.add_argument(
    '--time', type=int, default=4300,
    help='作业运行时间（分钟）')


class Trainer:
    """
    评估器类 —— 封装一次评估作业

    与预训练的Trainer结构相同，但调用的是eval_main而非app_main。
    submitit通过__call__执行评估，通过checkpoint()处理作业抢占。
    """

    def __init__(self, args_eval=None, resume_preempt=None):
        self.eval_name = args_eval['eval_name']  # 评估类型（如'image_classification_frozen'）
        self.args_eval = args_eval                # 评估配置参数
        self.resume_preempt = resume_preempt      # 是否从抢占恢复

    def __call__(self):
        """submitit 执行入口 —— 启动评估"""
        eval_name = self.eval_name
        args_eval = self.args_eval
        resume_preempt = self.resume_preempt

        logger.info('loaded eval params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args_eval)

        # 启动评估（eval_name决定评估类型）
        eval_main(
            eval_name,
            args_eval=args_eval,
            resume_preempt=resume_preempt)

    def checkpoint(self):
        """
        作业被抢占时的处理

        返回DelayedSubmission让submitit自动重新排队作业。
        resume_preempt=True表示恢复时将load_checkpoint设为True。
        """
        fb_trainer = Trainer(self.args_eval, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_evals_with_parsed_args(
    args_for_evals,      # 评估配置列表
    submitit_folder,     # submitit日志目录
    partition='learnlab,learnfair',  # SLURM分区（默认支持两个分区）
    timeout=4300,        # 超时时间（分钟）
    nodes=1,             # 节点数
    tasks_per_node=1,    # 每节点GPU数
    delay_seconds=10,    # 提交延迟（避免同时提交过多作业）
    exclude_nodes=None   # 排除节点列表
):
    """
    使用 submitit 向 SLURM 提交评估作业

    注意 delay_seconds 参数：
    多个评估作业会错开提交，避免同时冲击SLURM调度器。
    """
    if not isinstance(args_for_evals, list):
        logger.info(f'Passed in eval-args of type {type(args_for_evals)}')
        args_for_evals = [args_for_evals]

    # 延迟提交（避免作业同时冲击调度器）
    time.sleep(delay_seconds)
    logger.info('Launching evaluations in separate jobs...')

    executor = submitit.AutoExecutor(
        folder=os.path.join(submitit_folder, 'job_%j'),
        slurm_max_num_timeout=20)

    executor.update_parameters(
        slurm_partition=partition,
        slurm_mem_per_gpu='55G',
        timeout_min=timeout,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=12,
        gpus_per_node=tasks_per_node)

    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    # 批量提交评估作业
    jobs, trainers = [], []
    with executor.batch():
        for ae in args_for_evals:
            fb_trainer = Trainer(ae)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        logger.info(f'Launched eval job with id {job.job_id}')


def launch_evals():
    """
    启动分布式评估的主函数

    与预训练版本的launch()结构相同：
    1. 准备配置文件列表（单个或批量）
    2. 解析每个YAML配置文件
    3. 提取节点数和GPU数
    4. 提交submitit作业
    """
    # 步骤1：准备配置文件列表
    config_fnames = [args.fname]
    if args.batch_launch:
        # 批量启动：fname指向的文件包含多个配置文件路径
        with open(args.fname, 'r') as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)

    # 步骤2：解析配置文件
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

    # 步骤3：提交评估作业
    launch_evals_with_parsed_args(
        args_for_evals=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude)


if __name__ == '__main__':
    args = parser.parse_args()
    launch_evals()
