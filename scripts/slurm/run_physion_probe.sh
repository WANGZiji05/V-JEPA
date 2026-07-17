#!/bin/bash
# ============================================================================
# Physion++ Attentive Probe 训练 —— SLURM 作业提交脚本
# ============================================================================
#
# 【作业说明】
#   在 Physion++ 数据集上运行 V-JEPA attentive probe 评估：
#   - 冻结 V-JEPA 预训练编码器
#   - 在 readout_data 上训练 AttentiveClassifier 探针
#   - 在 test_data 上评估探针的 OCP（物体接触预测）准确率
#   - 对 4 种物理属性 (mass, friction, elasticity, deformability) 独立评估
#
# 【使用方法】
#   sbatch scripts/slurm/run_physion_probe.sh
#
# 【自定义参数】
#   可以使用环境变量覆盖默认值：
#     CONFIG=configs/evals/physion_attentive_probe.yaml sbatch scripts/slurm/run_physion_probe.sh
#     PRETRAIN_FOLDER=/path/to/models sbatch scripts/slurm/run_physion_probe.sh
# ============================================================================

# ============================================================================
# SLURM 作业配置
# ============================================================================
#SBATCH --job-name=physion_probe          # 作业名称
#SBATCH --gres=gpu:1                      # 【关键】申请 1 块 GPU
#SBATCH --ntasks=1                        # 任务数（单 GPU 用 1）
#SBATCH --cpus-per-task=8                 # 每个任务所需的 CPU 核心数
#SBATCH --time=04:00:00                   # 预估运行时间 (时:分:秒)
#                                          # Physion++ 数据集较小，单属性约 10-20 分钟
#                                          # 4 种属性串行约 1-2 小时，预留 4 小时余量
#SBATCH --mem=32G                         # 申请内存大小
#SBATCH --output=logs/physion_probe_%j.out  # 标准输出日志
#SBATCH --error=logs/physion_probe_%j.err   # 错误输出日志

# ============================================================================
# 环境设置
# ============================================================================

# 创建日志目录
mkdir -p logs

echo "=============================================="
echo "Physion++ Attentive Probe Evaluation"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# 加载必要的软件环境（根据你的集群配置调整）
# module load cuda/11.8
# module load python/3.10
# module load ffmpeg  # decord 视频解码可能需要

# 激活 Python 虚拟环境（根据你的环境调整）
# source /path/to/your/venv/bin/activate
# conda activate vjepa

# ============================================================================
# 配置参数（可通过环境变量覆盖）
# ============================================================================

# 配置文件路径
CONFIG="${CONFIG:-configs/evals/physion_attentive_probe.yaml}"

# 预训练模型路径
PRETRAIN_FOLDER="${PRETRAIN_FOLDER:-/research/d7/spc/yrwang5/V-JEPA/pretrained_models/}"

# 预训练检查点
CHECKPOINT="${CHECKPOINT:-jepa-latest.pth.tar}"

# Physion++ 数据 CSV 文件路径
TRAIN_CSV="${TRAIN_CSV:-/research/d7/spc/yrwang5/V-JEPA/V-JEPA/physion_csv/readout_data.csv}"
VAL_CSV="${VAL_CSV:-/research/d7/spc/yrwang5/V-JEPA/V-JEPA/physion_csv/test_data.csv}"

# 评估模式: per_property, joint
EVAL_MODE="${EVAL_MODE:-per_property}"

# 单个属性评估（留空则评估全部 4 种属性）
# 可选值: mass, friction, elasticity, deformability
PROPERTY="${PROPERTY:-}"

# ============================================================================
# 运行时配置
# ============================================================================

echo ""
echo "Configuration:"
echo "  Config file:     ${CONFIG}"
echo "  Pretrain folder: ${PRETRAIN_FOLDER}"
echo "  Checkpoint:      ${CHECKPOINT}"
echo "  Train CSV:       ${TRAIN_CSV}"
echo "  Val CSV:         ${VAL_CSV}"
echo "  Eval mode:       ${EVAL_MODE}"
echo "  Property:        ${PROPERTY:-all}"
echo ""

# ============================================================================
# 运行评估
# ============================================================================

echo "Starting evaluation..."
echo "Start time: $(date)"

python -m evals.main \
    --fname "${CONFIG}" \
    --devices cuda:0 \
    pretrain.folder="${PRETRAIN_FOLDER}" \
    pretrain.checkpoint="${CHECKPOINT}" \
    data.dataset_train="${TRAIN_CSV}" \
    data.dataset_val="${VAL_CSV}" \
    eval_mode="${EVAL_MODE}" \
    ${PROPERTY:+properties="[${PROPERTY}]"}

# 说明：上面的 ${PROPERTY:+properties="[${PROPERTY}]"} 是 bash 的
# "如果变量非空则替换" 语法。如果 $PROPERTY 为空，这行不会输出任何内容。

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Evaluation completed"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
