# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
================================================================================
V-JEPA 日志与监控工具 (Logging & Monitoring Utilities)
================================================================================

本模块为 V-JEPA 训练过程提供日志记录、性能计时和统计监控功能。

在大规模训练中，监控系统状态至关重要 —— 你需要知道：
  - 每个 step 花了多长时间？（性能瓶颈在哪里？）
  - 损失函数的当前值和历史趋势如何？
  - 梯度的范数是否正常？（有没有梯度爆炸或消失？）
  - AdamW 优化器的动量缓冲区状态如何？

本模块提供的功能：
  1. gpu_timer()      — 精确计时 GPU 操作的执行时间（利用 CUDA Event）
  2. get_logger()     — 获取统一格式的 Python 日志记录器
  3. CSVLogger        — 将训练指标记录到 CSV 文件（方便后续用 Excel / Python 分析）
  4. AverageMeter     — 在线计算平均值、最大值、最小值（无需存储所有历史数据）
  5. grad_logger()    — 统计模型梯度的范数（诊断梯度健康）
  6. adamw_logger()   — 统计 AdamW 优化器的动量缓冲区状态
================================================================================
"""

import logging
import sys

import torch


def gpu_timer(closure, log_timings=True):
    """
    使用 CUDA Event 精确计时 GPU 操作的执行时间。

    为什么需要特殊的方法计时？
        普通的 Python 计时（如 time.time()）只能测量 CPU 墙上时间。
        但 GPU 操作是异步的 —— CPU 发出指令后不等 GPU 完成就继续执行。
        因此 time.time() 可能返回的时间远小于 GPU 实际执行时间。

        CUDA Event 在 GPU 流中插入时间戳，可以精确测量 GPU kernel 的执行时间。

    参数：
        closure:     一个无参数的函数（闭包），包含需要计时的 GPU 操作
        log_timings: 是否启用计时（如果为 False 或 CUDA 不可用，跳过计时）
                     仅在主进程（rank=0）中启用，避免所有 GPU 都做同步

    返回值：
        (result, elapsed_time) 元组：
          - result:       closure() 的返回值
          - elapsed_time: GPU 操作的执行时间（毫秒），如果未计时则为 -1.0

    使用示例：
        def my_forward():
            return model(data)

        output, elapsed_ms = gpu_timer(my_forward)
        print(f"前向传播耗时: {elapsed_ms:.2f} ms")
    """

    # 只有在 CUDA 可用且明确要求计时时才启用
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.  # 默认值，表示未计时
    if log_timings:
        # CUDA Event 是 GPU 侧的时间戳对象
        # enable_timing=True 使其记录精确的时间信息
        start = torch.cuda.Event(enable_timing=True)  # 开始事件
        end = torch.cuda.Event(enable_timing=True)    # 结束事件
        start.record()  # 在 GPU 流中插入开始时间戳

    # 执行真正的操作
    result = closure()

    if log_timings:
        end.record()                          # 在 GPU 流中插入结束时间戳
        torch.cuda.synchronize()              # 等待 GPU 完成所有操作（阻塞 CPU）
        elapsed_time = start.elapsed_time(end) # 计算两个事件之间的时间差（毫秒）

    return result, elapsed_time


# 日志输出的格式模板
# [LEVEL   ][时间戳         ][函数名                     ] 消息内容
# 例如：
#   [INFO    ][2024-01-15 10:30:45][train                    ] Epoch 1, Step 100/1000
LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    """
    获取统一格式的 Python 日志记录器。

    参数：
        name:  日志记录器的名称（通常传入 __name__ 来区分不同模块的日志）
        force: 是否强制重新配置 basicConfig（Python 3.8+ 支持）

    返回值：
        logging.Logger 对象

    使用示例：
        logger = get_logger(__name__)
        logger.info("训练开始")
        logger.warning("GPU 内存不足，减小 batch size")
    """
    logging.basicConfig(
        stream=sys.stdout,       # 输出到标准输出（即控制台）
        level=logging.INFO,      # 最低日志级别为 INFO（DEBUG 级别的消息会被忽略）
        format=LOG_FORMAT,       # 使用上面定义的格式
        datefmt=DATE_FORMAT,      # 时间戳格式
        force=force              # 强制覆盖已有的 basicConfig（如果需要）
    )
    return logging.getLogger(name=name)


class CSVLogger(object):
    """
    将训练指标（损失、学习率等）以 CSV 格式写入文件。

    为什么用 CSV 而不是 TensorBoard 或其他工具？
        CSV 是最通用、最简单的格式，可以被 Excel、pandas、matplotlib 等
        任何工具读取和可视化，不依赖特定的可视化平台。

    CSV 文件结构：
        第一行是表头（列名），后续每一行是一次 step 的记录。
        例如：
            step,loss,learning_rate,elapsed_time
            0,2.345,0.001,123.4
            1,2.201,0.001,115.6
            ...

    参数：
        fname: CSV 文件的路径
        *argv: 可变参数，每个元素是一个 (格式, 列名) 的元组
               例如: ('%d', 'step'), ('%.4f', 'loss'), ('%f', 'lr')
    """

    def __init__(self, fname, *argv):
        # 保存文件路径
        self.fname = fname

        # 存储每个列的格式化字符串（如 '%d', '%.4f'）
        self.types = []

        # --- 写入表头 (Header) ---
        # '+a' 模式：以追加模式打开，如果文件不存在则创建
        with open(self.fname, '+a') as f:
            # enumerate(argv, 1) 从索引 1 开始
            for i, v in enumerate(argv, 1):
                # v 是 (格式, 列名) 的元组，v[0] 是格式，v[1] 是列名
                self.types.append(v[0])
                if i < len(argv):
                    # 不是最后一列：列名后加逗号分隔
                    print(v[1], end=',', file=f)
                else:
                    # 最后一列：列名后加换行
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        """
        向 CSV 文件追加一行记录。

        参数：
            *argv: 与 __init__ 中的列一一对应的值

        例如：
            logger.log(100, 2.345, 0.001)
            → 写入: 100,2.3450,0.001000
        """
        with open(self.fname, '+a') as f:
            # zip: 将格式和值配对
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                # tv[0] 是格式字符串（如 '%.4f'），tv[1] 是值（如 2.345）
                # 使用格式字符串格式化值：tv[0] % tv[1]
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """
    在线计算并存储统计指标：当前值、平均值、最大值、最小值。

    为什么需要"在线"计算？
        训练过程中有成千上万个 step，不能把所有值都存在内存中再算统计。
        AverageMeter 使用了递推更新公式，每次只需 O(1) 的时间和空间。

    统计量说明：
        val:  最近一次更新的值（当前值）
        avg:  所有更新值的加权平均数
        max:  所有更新值的最大值
        min:  所有更新值的最小值
        sum:  所有更新值的加权总和
        count: 总的更新次数（用于加权）

    加权平均：
        当 batch size 不一致时，可以通过 n 参数进行加权。
        例如：
            meter.update(0.5, n=64)  表示 batch size=64，loss=0.5
            meter.update(0.3, n=32)  表示 batch size=32，loss=0.3
            平均 loss = (0.5*64 + 0.3*32) / (64+32) = 0.433
    """

    def __init__(self):
        # 初始化时重置所有统计量
        self.reset()

    def reset(self):
        """ 将所有统计量重置为零/无穷大 """
        self.val = 0                 # 当前值
        self.avg = 0                 # 加权平均值
        self.max = float('-inf')     # 最大值（初始化为负无穷，确保任何值都是"更大"的）
        self.min = float('inf')      # 最小值（初始化为正无穷，确保任何值都是"更小"的）
        self.sum = 0                 # 加权总和 = sum(val_i * n_i)
        self.count = 0               # 加权计数 = sum(n_i)

    def update(self, val, n=1):
        """
        更新统计量。

        参数：
            val: 新观测到的值
            n:   权重（通常是这个值对应的样本数，如 batch size）
        """
        # 更新当前值
        self.val = val

        # 更新最大值和最小值
        # try-except 是为了处理某些无法比较的特殊值（如包含 NaN 的张量）
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass  # 如果比较失败，保留原来的值

        # 递推更新总和与计数
        self.sum += val * n   # 加权求和
        self.count += n       # 累加权重

        # 加权平均 = 加权总和 / 权重总和
        self.avg = self.sum / self.count


def grad_logger(named_params):
    """
    收集模型梯度的统计信息（用于诊断训练状态）。

    只统计非 bias、非一维参数的梯度范数：
      - 排除 bias：bias 的梯度通常很小，会拉低平均值
      - 排除一维参数：通常是 LayerNorm 的 weight/bias，也会干扰统计

    参数：
        named_params: 模型的 named_parameters() 迭代器
                      每个元素是 (参数名称, 参数张量) 的元组

    返回值：
        stats: AverageMeter 对象，额外附加了两个属性：
          - stats.first_layer: "第一个" QKV 层的梯度范数（实际是最后一个遇到的 qkv 层）
          - stats.last_layer:  "最后一个" QKV 层的梯度范数

    用途：
        监控 first_layer 和 last_layer 的梯度范数可以诊断：
        - 梯度消失：first_layer 接近 0 → 信号在反向传播中衰减殆尽
        - 梯度爆炸：任意层的梯度过大 → 需要梯度裁剪 (gradient clipping)
    """
    stats = AverageMeter()
    stats.first_layer = None   # 记录第一个遇到的 qkv 层的梯度范数
    stats.last_layer = None    # 记录最后一个遇到的 qkv 层的梯度范数

    for n, p in named_params:
        # 跳过没有梯度的参数（如冻结的层）
        # 跳过 bias 参数（名称以 '.bias' 结尾）
        # 跳过一维参数（如 LayerNorm 的 weight/bias）
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            # 计算梯度的 L2 范数（所有元素的平方和再开根号）
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)

            # 如果参数名包含 'qkv'，记录其梯度范数
            # 第一个遇到的 qkv 层 → stats.first_layer
            # 最后一个遇到的 qkv 层 → stats.last_layer（不断被覆盖）
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm

    # 如果没有找到任何 qkv 层，设 first_layer 和 last_layer 为 0
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.

    return stats


def adamw_logger(optimizer):
    """
    收集 AdamW 优化器的动量缓冲区统计信息。

    AdamW 优化器基本概念：
        AdamW 维护两个动量缓冲区：
          - exp_avg   (一阶矩估计)：类似"速度"，表示梯度的滑动平均
          - exp_avg_sq (二阶矩估计)：类似"加速度"，表示梯度平方的滑动平均

        这两个缓冲区的绝对值大小反映了优化器的"动量状态"：
          - 值太小 → 优化器没有积累足够的动量，训练可能刚开始
          - 值太大 → 可能面临梯度爆炸的风险
          - exp_avg_sq 过大 → 自适应学习率变得极小，参数几乎不再更新

    参数：
        optimizer: AdamW 优化器对象

    返回值：
        dict: {'exp_avg': AverageMeter, 'exp_avg_sq': AverageMeter}
              包含了所有参数组的一阶和二阶矩缓冲区的平均绝对值
    """
    # 获取优化器的完整状态字典
    # state_dict() 返回包含 'state' 键的字典
    # 'state' 是一个 dict，键是参数 ID，值是该参数对应的优化器状态
    state = optimizer.state_dict().get('state')

    # 为两种动量缓冲区各创建一个 AverageMeter
    exp_avg_stats = AverageMeter()      # 一阶矩的统计
    exp_avg_sq_stats = AverageMeter()    # 二阶矩的统计

    for key in state:
        s = state.get(key)

        # 一阶矩缓冲区：对其所有元素的绝对值取平均
        # .abs() → 取绝对值
        # .mean() → 求平均
        # 这衡量了梯度的平均信号强度
        exp_avg_stats.update(float(s.get('exp_avg').abs().mean()))

        # 二阶矩缓冲区：同样取绝对值的平均
        # 二阶矩的值通常比一阶矩小（因为是梯度的平方的滑动平均）
        exp_avg_sq_stats.update(float(s.get('exp_avg_sq').abs().mean()))

    return {'exp_avg': exp_avg_stats, 'exp_avg_sq': exp_avg_sq_stats}
