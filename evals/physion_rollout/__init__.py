# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
V-JEPA Physion++ Rollout Evaluation Module

评估 V-JEPA 在 Physion++ 上的 latent dynamics 预测能力。
直接测试 JEPA 核心机制：给定前段视频的 latent 特征，
预测后段视频的 latent 特征，在特征空间（非像素空间）度量预测质量。
"""
