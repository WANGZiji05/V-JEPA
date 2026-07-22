#!/bin/bash
# ============================================================================
# Physion++ Latent Rollout 评估 —— SLURM 作业提交脚本
# ============================================================================
#SBATCH --job-name=physion_rollout
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/physion_rollout_%j.out
#SBATCH --error=logs/physion_rollout_%j.err

mkdir -p logs

echo "=============================================="
echo "V-JEPA Physion++ Latent Rollout Evaluation"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "=============================================="

# 配置
CONFIG="${CONFIG:-configs/evals/physion_rollout.yaml}"
PRETRAIN_FOLDER="${PRETRAIN_FOLDER:-/your_path_to_pretrained_models/}"
CHECKPOINT="${CHECKPOINT:-jepa-latest.pth.tar}"
TEST_CSV="${TEST_CSV:-/your_path_to_physion_csv/test_data.csv}"
DATA_ROOT="${DATA_ROOT:-/your_path_to_physion_data_root/}"

echo "Config:       ${CONFIG}"
echo "Pretrain dir: ${PRETRAIN_FOLDER}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Test CSV:     ${TEST_CSV}"
echo "Data root:    ${DATA_ROOT}"

python -m evals.main \
    --fname "${CONFIG}" \
    --devices cuda:0

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
exit ${EXIT_CODE}
