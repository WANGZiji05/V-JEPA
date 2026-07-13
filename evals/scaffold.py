# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ============================================================================
# 评估脚手架 —— 动态加载评估模块
# ============================================================================
# 与 app/scaffold.py 类似，但是用于评估任务。
# 根据 eval_name 参数动态导入对应的评估模块。
#
# eval_name 的取值：
#   'image_classification_frozen' → evals.image_classification_frozen.eval
#   'video_classification_frozen' → evals.video_classification_frozen.eval

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(
    eval_name,         # 评估任务名称（如 'image_classification_frozen'）
    args_eval,         # 从配置文件解析的参数字典
    resume_preempt=False  # 是否从集群抢占恢复
):
    """
    动态加载并启动指定的评估任务

    工作原理：
    1. 根据eval_name构造模块路径：f'evals.{eval_name}.eval'
    2. 动态导入该模块
    3. 调用模块的 main() 函数
    """
    logger.info(f'Running evaluation: {eval_name}')
    return importlib.import_module(f'evals.{eval_name}.eval').main(
        args_eval=args_eval,
        resume_preempt=resume_preempt)
