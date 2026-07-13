# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
================================================================================
V-JEPA 学习率与权重衰减调度器 (Learning Rate & Weight Decay Schedulers)
================================================================================

本模块在 V-JEPA 中负责 **动态调整训练过程中的两个关键超参数**：

1. **学习率 (Learning Rate, LR)**：控制模型参数每次更新的步长。
   太大 → 训练不稳定、损失发散；太小 → 收敛过慢。
2. **权重衰减 (Weight Decay, WD)**：一种正则化手段，防止模型过拟合。
   本质是每次更新时把参数缩小一点点（等价于 L2 正则）。

为什么需要"调度器"而不是固定值？
  因为训练的不同阶段对这两个参数的需求不同：
  - 训练初期：需要从小学习率开始"热身"，避免模型被随机初始化带偏。
  - 训练中期：需要较大的学习率让模型充分学习。
  - 训练后期：需要逐渐降低学习率，让模型精细收敛。

V-JEPA 使用 **余弦调度策略 (Cosine Schedule)** 作为衰减曲线：
  - 值从某个参考值开始，按余弦曲线平滑地衰减到最终值。
  - 学习率还额外包含一个 **线性热身 (Warmup)** 阶段。

本模块包含两个类：
  - WarmupCosineSchedule：管理学习率的"热身 + 余弦衰减"
  - CosineWDSchedule：管理权重衰减的"纯余弦衰减"（无热身）
================================================================================
"""

import math


class WarmupCosineSchedule(object):
    """
    学习率调度器：线性热身 → 余弦衰减 (Linear Warmup + Cosine Decay)

    学习率变化曲线：
        Step 0 ~ warmup_steps:
            LR 从 start_lr 线性增长到 ref_lr （热身阶段）
        Step warmup_steps ~ T_max:
            LR 从 ref_lr 按余弦曲线衰减到 final_lr （衰减阶段）

    直观理解：
        想象一辆车从静止（start_lr）慢慢加速到巡航速度（ref_lr），
        然后逐渐减速（余弦衰减）直到接近停下来（final_lr）。

    关键参数：
        optimizer:     PyTorch 优化器对象，调度器会直接修改其内部的 lr
        warmup_steps:  热身步数，前 N 步学习率线性上升
        start_lr:      训练第 0 步的学习率（通常很小，如 1e-6）
        ref_lr:        参考学习率，热身结束后达到的值（通常是最优 LR）
        T_max:         总训练步数（包含热身步数）
        final_lr:      训练结束时的学习率（默认为 0）
        last_epoch:    上次训练的 epoch 数（用于恢复训练，默认 -1 表示从头开始）
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        # 保存 PyTorch 优化器的引用，后续通过它修改每个参数组的 lr
        self.optimizer = optimizer

        # 记录学习率的三个关键值
        self.start_lr = start_lr    # 热身起始学习率
        self.ref_lr = ref_lr        # 参考学习率（热身结束后的峰值）
        self.final_lr = final_lr    # 最终学习率（余弦衰减的终点）

        # 记录热身步数和总衰减步数
        # T_max 需要减去 warmup_steps，因为余弦衰减只在热身结束后才开始
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps

        # 内部步数计数器，每次调用 step() 时自增
        self._step = 0.

    def step(self):
        """
        执行一步调度：步数 +1，计算新的学习率，并更新到优化器中。

        每次优化器更新参数之前调用此方法（通常在每个训练 step 的开始）。
        调用频率取决于训练代码的设计，可以是每 step 一次，
        也可以是每 epoch 一次（此时需要重新设计逻辑）。

        返回值：
            new_lr: 计算出的新学习率（float），用于日志记录
        """
        self._step += 1

        # --- 阶段一：线性热身 (Linear Warmup) ---
        if self._step < self.warmup_steps:
            # progress: 0.0 → ~1.0，表示热身完成的比例
            progress = float(self._step) / float(max(1, self.warmup_steps))
            # 线性插值：从 start_lr 平滑过渡到 ref_lr
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)

        # --- 阶段二：余弦衰减 (Cosine Annealing) ---
        else:
            # progress: 0.0 → 1.0，表示衰减完成的比例
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            # 余弦函数在 [0, pi] 区间从 1 平滑降到 -1
            # 将它映射为: (1+cos)/2 → 从 1 平滑降到 0
            # 这个值 (0→1) 再与 ref_lr 和 final_lr 做插值
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr)
                * 0.5
                * (1. + math.cos(math.pi * progress))
            )
            # max() 确保学习率不会低于 final_lr（防止余弦函数因浮点误差略微低于目标）

        # 将新学习率写入优化器的每个参数组
        # optimizer.param_groups 是一个 list，每个元素是一个 dict，包含 'lr', 'params' 等键
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):
    """
    权重衰减调度器：纯余弦衰减 (Cosine Decay，无热身)

    权重衰减变化曲线：
        Step 0 ~ T_max:
            WD 从 ref_wd 按余弦曲线衰减到 final_wd

    直观理解：
        权重衰减类似于"遗忘"——让参数值逐步向 0 收缩。
        训练初期，较强的权重衰减提供正则化效果；
        训练后期，衰减减弱，让模型更精细地拟合数据。

    与 WarmupCosineSchedule 的区别：
        1. 没有热身阶段，从第 0 步就开始余弦衰减
        2. 它修改的是 optimizer.param_groups 中的 'weight_decay' 字段，而非 'lr'
        3. 支持"排除组" (WD_exclude)：某些参数组可以跳过 WD 调整

    关键参数：
        optimizer: PyTorch 优化器对象
        ref_wd:     参考权重衰减值（训练开始时的 WD）
        T_max:      总训练步数
        final_wd:   训练结束时的权重衰减值（默认为 0）
    """

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        # 保存优化器引用
        self.optimizer = optimizer

        # 权重衰减的起点和终点
        self.ref_wd = ref_wd      # 参考权重衰减
        self.final_wd = final_wd  # 最终权重衰减

        # 总步数（不需要减去热身，因为没有热身阶段）
        self.T_max = T_max

        # 内部步数计数器
        self._step = 0.

    def step(self):
        """
        执行一步调度：步数 +1，计算新的权重衰减，并更新到优化器中。

        返回值：
            new_wd: 计算出的新权重衰减值（float），用于日志记录
        """
        self._step += 1

        # progress: 0.0 → 1.0，表示训练完成的比例
        progress = self._step / self.T_max

        # 余弦衰减公式：与学习率调度器相同
        # final_wd + (ref_wd - final_wd) * 0.5 * (1 + cos(pi * progress))
        # 在 progress=0 时，值为 ref_wd；在 progress=1 时，值为 final_wd
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        # 边界保护：确保 WD 不会越过 final_wd
        # - 如果 final_wd <= ref_wd（WD 逐渐减小）：确保 new_wd >= final_wd
        # - 如果 final_wd >  ref_wd（WD 逐渐增大）：确保 new_wd <= final_wd
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        # 将新权重衰减写入优化器的每个参数组
        # 注意：检查 'WD_exclude' 标志 —— 被标记排除的参数组不参与 WD 调整
        # 这允许某些参数（如 bias、LayerNorm 的 weight）不受权重衰减约束
        for group in self.optimizer.param_groups:
            # 'WD_exclude' 如果不存在或为 False，则更新该组的 weight_decay
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd

        return new_wd
