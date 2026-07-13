# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 应用脚手架 —— 动态加载和启动训练/评估模块
# ============================================================================
# 这个文件是一个轻量级的"路由"层。
# 根据配置文件中的 'app' 字段，动态导入对应的训练模块。
#
# 例如：
# - app='vjepa' → 导入 app.vjepa.train → 运行预训练
# - 未来可能支持其他训练范式（如其他自监督方法）
#
# 这个设计使代码结构更灵活，可以在不修改入口文件的情况下
# 添加新的训练任务。

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):
    """
    动态加载并启动指定的训练/评估任务

    参数:
        app: 任务名称（如 'vjepa'）
        args: 从配置文件解析的参数
        resume_preempt: 是否从集群抢占恢复

    实际执行：
    1. 根据app名称构造模块路径：f'app.{app}.train'
    2. 动态导入该模块（如 app.vjepa.train）
    3. 调用该模块的 main() 函数
    """
    logger.info(f'Running pre-training of app: {app}')
    # importlib.import_module 动态导入模块
    # 相当于：from app.vjepa.train import main as train_main
    #         train_main(args=args, resume_preempt=resume_preempt)
    return importlib.import_module(f'app.{app}.train').main(
        args=args,
        resume_preempt=resume_preempt)
