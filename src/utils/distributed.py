# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
================================================================================
V-JEPA 分布式训练工具 (Distributed Training Utilities)
================================================================================

本模块为 V-JEPA 提供跨多 GPU 和多节点的分布式训练支持。

分布式训练的核心动机：
  V-JEPA 是一个大规模视频理解模型，训练它需要处理海量视频数据。
  单个 GPU 的显存和计算能力远远不够，因此需要：
    - 数据并行 (Data Parallelism)：多张 GPU 各自处理不同的数据片段，
      然后在反向传播时同步梯度。
    - 多节点训练：跨多台机器的 GPU 协同工作（通常通过 SLURM 集群管理）。

PyTorch 的 torch.distributed 提供了底层的通信原语（broadcast、all_reduce 等），
本模块在此之上封装了 V-JEPA 需要的便捷功能。

本模块包含：
  1. init_distributed()     — 初始化分布式环境（支持 SLURM 自动检测）
  2. AllGather              — 跨 GPU 收集张量（自定义 autograd 操作）
  3. AllReduceSum           — 跨 GPU 求和（自定义 autograd 操作）
  4. AllReduce              — 跨 GPU 求平均（自定义 autograd 操作）

自定义 autograd 操作的必要性：
  普通的 all_gather / all_reduce 在前向传播时正常工作，但如果不实现 backward，
  PyTorch 的自动求导会在反向传播时中断。因此我们用 torch.autograd.Function
  来实现它们，确保梯度也能在 GPU 之间正确同步。
================================================================================
"""

import os

import torch
import torch.distributed as dist

from logging import getLogger

# 获取日志记录器，用于输出初始化过程中的信息
logger = getLogger()


def init_distributed(port=37123, rank_and_world_size=(None, None)):
    """
    初始化 PyTorch 分布式训练环境。

    此函数负责配置分布式通信所需的全部环境变量和进程组。
    设计上具有容错性：如果分布式环境不可用（如在单 GPU 机器上调试），
    会自动退化为单 GPU 模式 (rank=0, world_size=1)。

    参数：
        port:                 用于分布式通信的 TCP 端口号（默认为 37123）
        rank_and_world_size:  (rank, world_size) 元组，手动指定。
                              如果为 (None, None)，则尝试从 SLURM 环境变量自动检测。

    返回值：
        (world_size, rank) 元组：
          - world_size: 全局 GPU 总数（所有机器上的 GPU 总和）
          - rank:       当前进程的序号（0 到 world_size-1，唯一标识）

    SLURM 集成：
        当在集群上通过 SLURM 提交任务时，SLURM 会自动设置以下环境变量：
          - SLURM_NTASKS：任务总数（= world_size）
          - SLURM_PROCID：当前任务 ID（= rank）
          - HOSTNAME：    当前机器的主机名（用作 MASTER_ADDR）

    通信后端：
        NCCL (NVIDIA Collective Communications Library) 是 NVIDIA GPU 专用的
        高性能集合通信库，是分布式 GPU 训练的首选后端。
    """

    # 如果已经初始化过，直接返回当前的 world_size 和 rank
    # 这防止了重复初始化导致的错误
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    # 解包手动指定的 rank 和 world_size
    rank, world_size = rank_and_world_size

    # 设置主节点地址为 localhost（单机多卡时的默认值）
    os.environ['MASTER_ADDR'] = 'localhost'

    # --- 如果没有手动指定，尝试从 SLURM 环境变量自动检测 ---
    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])    # SLURM 任务总数 = 总 GPU 数
            rank = int(os.environ['SLURM_PROCID'])          # 当前进程 ID
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']  # 当前机器名作为主节点
        except Exception:
            # SLURM 环境变量不存在 → 判定为单 GPU 环境
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        # 设置通信端口
        os.environ['MASTER_PORT'] = str(port)

        # 初始化进程组 —— 这是分布式训练的核心一步
        # world_size: 所有参与训练的进程数（每个 GPU 一个进程）
        # rank:       当前进程的全局编号
        torch.distributed.init_process_group(
            backend='nccl',          # 使用 NCCL 作为 GPU 通信后端
            world_size=world_size,
            rank=rank
        )
    except Exception as e:
        # 如果初始化失败（例如端口被占用），退化为单 GPU 模式
        world_size, rank = 1, 0
        logger.info(f'Rank: {rank}. Distributed training not available {e}')

    return world_size, rank


class AllGather(torch.autograd.Function):
    """
    自定义 autograd 操作：跨所有 GPU 收集张量（All-Gather）。

    作用：
        假设有 4 张 GPU，每张 GPU 上有一个形状为 [B, D] 的张量。
        All-Gather 将所有 GPU 上的张量在第 0 维拼接，得到 [4*B, D]，
        并将这个结果广播到所有 GPU。

    在 V-JEPA 中的用途：
        V-JEPA 的对比学习中，需要计算跨 GPU 的全局负样本池。
        A 卡不知道 B 卡上的样本，AllGather 让每张卡都看到所有样本的表示。

    前向传播 (forward)：
        将每张 GPU 上的张量收集并拼接，返回给所有 GPU。

    反向传播 (backward)：
        梯度的反向操作：全局梯度先被 all_reduce（求和），
        然后按 rank 切片，每张 GPU 只保留属于自己那部分的梯度。
        这样保证了梯度在反向传播时的正确性。

    注意：
        如果只有 1 张 GPU（单卡模式），直接返回输入，不做任何通信。
    """

    @staticmethod
    def forward(ctx, x):
        # ctx 是一个上下文对象，用于在 forward 和 backward 之间传递信息
        # 单卡模式下不需要任何通信，直接返回
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            # contiguous() 确保张量在内存中是连续存储的（NCCL 通信要求）
            x = x.contiguous()
            # 为每张 GPU 创建一个与 x 相同形状的空张量，用于接收数据
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            # all_gather：将各 GPU 上的数据收集到 outputs 列表中
            dist.all_gather(outputs, x)
            # 在第 0 维拼接所有 GPU 的数据，形成全局张量
            return torch.cat(outputs, 0)
        # 单卡模式：直接返回输入
        return x

    @staticmethod
    def backward(ctx, grads):
        # 反向传播：grads 是上游传下来的全局梯度
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            # 计算当前 GPU 对应的梯度切片范围
            # 例如 4 张 GPU，grads 共 40 行：rank0 → [0:10], rank1 → [10:20], ...
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()       # 起始索引
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1) # 结束索引
            grads = grads.contiguous()
            # all_reduce：将各 GPU 上的梯度求和并同步到所有 GPU
            # 这一步是为了聚合来自不同 GPU 的梯度贡献
            dist.all_reduce(grads)
            # 只返回属于当前 GPU 的那部分梯度
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):
    """
    自定义 autograd 操作：跨所有 GPU 求和 (All-Reduce with SUM)。

    作用：
        将每张 GPU 上的张量按元素求和，结果同步到所有 GPU。

    示例：
        GPU 0: [2, 3]
        GPU 1: [5, 7]
        GPU 2: [1, 4]
        GPU 3: [3, 2]
        → 所有 GPU 得到: [11, 16]

    前向传播 (forward)：
        对所有 GPU 的张量做 all_reduce 求和。

    反向传播 (backward)：
        梯度直接透传（all_reduce 本身的梯度就是输入梯度，
        因为求和操作对每个输入的偏导都是 1）。

    注意：
        此操作不除以世界大小，与 AllReduce 不同。
    """

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            # all_reduce 默认的操作就是 SUM
            # 将各 GPU 上的 x 求和，结果同步到所有 GPU
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        # 求和操作的梯度是恒等映射，直接返回
        return grads


class AllReduce(torch.autograd.Function):
    """
    自定义 autograd 操作：跨所有 GPU 求平均 (All-Reduce with MEAN)。

    作用：
        将每张 GPU 上的张量求和后除以 world_size（GPU 数量），结果同步到所有 GPU。
        等价于跨 GPU 求平均值。

    示例：
        GPU 0: [2, 3]
        GPU 1: [5, 7]
        GPU 2: [1, 4]
        GPU 3: [3, 2]
        → 求和 [11, 16]，除以 4 → 所有 GPU 得到: [2.75, 4.0]

    与 AllReduceSum 的区别：
        AllReduceSum 只是求和，AllReduce 在此基础上除以 world_size，得到平均值。

    V-JEPA 中的用途：
        某些损失函数需要计算跨所有 GPU 的统计平均值（如动量更新时的同步）。

    前向传播 (forward)：
        先在本地除以 world_size，再做 all_reduce 求和。
        （数学上等于求和后除以 world_size，但这样实现可减少一次通信）

    反向传播 (backward)：
        梯度直接透传。
    """

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            # 先让本地张量除以 world_size
            x = x.contiguous() / dist.get_world_size()
            # 再对所有 GPU 求 all_reduce 求和
            # 效果等同于：(sum of all GPUs) / world_size = 平均值
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
