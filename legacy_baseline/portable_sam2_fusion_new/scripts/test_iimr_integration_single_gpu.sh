#!/bin/bash
# IIMR 单 GPU 集成测试脚本
# 用于验证 IIMR 组件在单 GPU 环境下的基本功能
# 单流卫星分支版本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/utils/iimr_test_utils.sh"

# === 基础参数 ===
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-5e-5}"
SEED="${SEED:-44}"
DRY_RUN="${DRY_RUN:-0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LOG_TERMINAL_ENV="${LOG_TERMINAL_ENV:-1}"
LOG_TERMINAL_ENV_INCLUDE_ALL="${LOG_TERMINAL_ENV_INCLUDE_ALL:-0}"

# === IIMR 核心开关 ===
# 注: 默认值为 -1 或 -1.0 的是哨兵值, 表示"不覆盖config, 使用config文件中的默认值"
#     显式设为 0/1/具体数值时才会覆盖config默认值

# IIMR 总开关: 1=启用迭代推理, 0=关闭(等同基线)
IIMR_ENABLED="${IIMR_ENABLED:-1}"
# 迭代次数: 每张图执行几次 memory attention 精炼, config默认3 [哨兵值 -1]
IIMR_NUM_ITERATIONS="${IIMR_NUM_ITERATIONS:--1}"
# Memory 残差融合系数: 0.0=完全替换, 0.3=70%原始+30%memory, 1.0=完全用memory, config默认0.3 [哨兵值 -1.0]
IIMR_MEMORY_RESIDUAL_WEIGHT="${IIMR_MEMORY_RESIDUAL_WEIGHT:--1.0}"
# P0: 前 N 个 epoch 关闭 IIMR（0 表示不延迟）[哨兵值 -1]
IIMR_ENABLE_AFTER_EPOCHS="${IIMR_ENABLE_AFTER_EPOCHS:--1}"
# P0: IIMR 启用后 residual warmup 的 epoch 数（0 表示关闭 warmup）[哨兵值 -1]
IIMR_RESIDUAL_WARMUP_EPOCHS="${IIMR_RESIDUAL_WARMUP_EPOCHS:--1}"
# P0: residual warmup 起点（通常设 0.0）[哨兵值 -1.0]
IIMR_RESIDUAL_START_WEIGHT="${IIMR_RESIDUAL_START_WEIGHT:--1.0}"
# ROI 空间位置编码: 为IIMR memory attention注入可学习的ROI相对位置编码, config默认1 [哨兵值 -1]
IIMR_USE_ROI_POS_ENCODING="${IIMR_USE_ROI_POS_ENCODING:--1}"
# 梯度隔离: 1=切断mask→backbone梯度(已确认导致segm退化,应设为0), config默认1 [哨兵值 -1]
IIMR_DETACH_MASK_PATH="${IIMR_DETACH_MASK_PATH:--1}"

# === IIMR 拓扑相关 ===
# 拓扑 token 注入 Memory KV: 从mask预测提取boundary/skeleton token拼入attention [哨兵值 -1]
IIMR_USE_TOPOLOGY_MEMORY="${IIMR_USE_TOPOLOGY_MEMORY:--1}"
# Geometry-aware KV: 在 memory attention KV 注入 boundary token（路径B）[哨兵值 -1]
IIMR_USE_GEOMETRY_AWARE_KV="${IIMR_USE_GEOMETRY_AWARE_KV:--1}"
# 拓扑 token 送入 Mask Decoder: 拓扑信息直接参与 mask 预测 (依赖 topology_memory) [哨兵值 -1]
IIMR_TOPOLOGY_TO_DECODER="${IIMR_TOPOLOGY_TO_DECODER:--1}"
# 训练时用 GT mask 提取拓扑 token(而非pred mask), 梯度截断 (依赖 topology_memory) [哨兵值 -1]
IIMR_USE_GT_TOPOLOGY_TOKENS="${IIMR_USE_GT_TOPOLOGY_TOKENS:--1}"
# GT 拓扑 token 仅在第一次迭代使用, 后续迭代切换为 pred (依赖 gt_topology_tokens) [哨兵值 -1]
IIMR_GT_TOPOLOGY_ONLY_FIRST_ITER="${IIMR_GT_TOPOLOGY_ONLY_FIRST_ITER:--1}"

# === IIMR 动态提示 ===
# 动态提示总开关: 根据预测不确定性自动生成正负样本点 [哨兵值 -1]
IIMR_USE_DYNAMIC_PROMPTING="${IIMR_USE_DYNAMIC_PROMPTING:--1}"

# === Dynamic Prompting 参数 ===
# 不确定性采样点数: 每次迭代生成多少个动态提示点, config默认4 [哨兵值 -1]
IIMR_NUM_UNCERTAINTY_POINTS="${IIMR_NUM_UNCERTAINTY_POINTS:--1}"
# 正负样本比例: 0.25 表示 25%正样本 75%负样本, config默认0.25 [哨兵值 -1.0]
IIMR_DYNAMIC_BALANCE_RATIO="${IIMR_DYNAMIC_BALANCE_RATIO:--1.0}"
# 采样点最小像素距离: 0=不限制, >0时避免采样点过于密集, config默认0 [哨兵值 -1]
IIMR_DYNAMIC_MIN_POINT_DISTANCE="${IIMR_DYNAMIC_MIN_POINT_DISTANCE:--1}"
# 从第几个迭代开始使用动态提示: 1=第二次迭代开始, config默认1 [哨兵值 -1]
IIMR_DYNAMIC_START_ITERATION="${IIMR_DYNAMIC_START_ITERATION:--1}"
# 动态点嵌入强度: 控制动态提示与原始提示的融合比例, config默认1.0 [哨兵值 -1.0]
IIMR_DYNAMIC_POINT_MIX="${IIMR_DYNAMIC_POINT_MIX:--1.0}"
# 动态提示融合模式: "add"=直接相加, "gated"=门控融合, 空字符串=不覆盖config
IIMR_DYNAMIC_FUSION_MODE="${IIMR_DYNAMIC_FUSION_MODE:-}"

# === 数据路径 ===
SAT_DATA_ROOT="${SAT_DATA_ROOT:-/data/wangcheng/dataset/unversity-big-after-without-negative/university-s-train}"
SAM2_CKPT="${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_large.pt}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SAM2_CKPT="${SAM2_CKPT}"

# 打印配置摘要
print_test_header "单 GPU 集成测试 (单流卫星分支)" "
GPU: ${GPU_ID}
Epochs: ${EPOCHS}
Batch Size: ${BATCH_SIZE}
Grad Accum Steps: ${GRAD_ACCUM_STEPS}
LR: ${LR}
Seed: ${SEED}
---
IIMR Enabled: ${IIMR_ENABLED}
Num Iterations: ${IIMR_NUM_ITERATIONS}
Memory Residual Weight: ${IIMR_MEMORY_RESIDUAL_WEIGHT}
Use Dynamic Prompting: ${IIMR_USE_DYNAMIC_PROMPTING}
Dynamic Start Iteration: ${IIMR_DYNAMIC_START_ITERATION}
Detach Mask Path: ${IIMR_DETACH_MASK_PATH}
配置: rsprompter_anchor_satS_v11_sam2_large_full.py
"

START_TIME=$(date +%s)

activate_env || exit 1

check_gpu_available "${GPU_ID}" || exit 1

LOG_DIR=$(setup_log_dir "single_gpu")
TIMESTAMP=$(get_timestamp)
LOG_FILE="${LOG_DIR}/test_gpu${GPU_ID}_${TIMESTAMP}.log"

format_switch() {
    local key="$1"
    local value="$2"
    local unset_marker="$3"
    if [ "${value}" = "${unset_marker}" ]; then
        echo "${key}=inherit(default)"
    else
        echo "${key}=${value}"
    fi
}

is_sensitive_env_key() {
    local key_upper
    key_upper=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    [[ "${key_upper}" == *"PASSWORD"* || \
       "${key_upper}" == *"PASSWD"* || \
       "${key_upper}" == *"TOKEN"* || \
       "${key_upper}" == *"SECRET"* || \
       "${key_upper}" == *"API_KEY"* || \
       "${key_upper}" == *"ACCESS_KEY"* || \
       "${key_upper}" == *"PRIVATE_KEY"* || \
       "${key_upper}" == *"COOKIE"* || \
       "${key_upper}" == *"CREDENTIAL"* ]]
}

write_terminal_env_snapshot() {
    if [ "${LOG_TERMINAL_ENV}" != "1" ]; then
        return
    fi

    {
        echo "[终端环境快照]"
        echo "shell: ${SHELL}"
        echo "pwd: $(pwd)"
        echo "user: $(whoami)"
        echo "hostname: $(hostname)"
        echo "conda_default_env: ${CONDA_DEFAULT_ENV:-N/A}"
        echo "python_bin: ${PYTHON_BIN:-N/A}"
        echo "python_version: $(${PYTHON_BIN:-python} --version 2>&1 || true)"
        echo "path: ${PATH}"
        echo "log_terminal_env_include_all: ${LOG_TERMINAL_ENV_INCLUDE_ALL}"
        echo "env_vars:"

        if [ "${LOG_TERMINAL_ENV_INCLUDE_ALL}" = "1" ]; then
            env | sort | while IFS='=' read -r key value; do
                if is_sensitive_env_key "${key}"; then
                    echo "  ${key}=***REDACTED***"
                else
                    echo "  ${key}=${value}"
                fi
            done
        else
            env | sort | grep -E '^(CUDA|CONDA|PYTHON|PYTORCH|HF_|TRANSFORMERS|NCCL|OMP|MKL|TORCH|IIMR_|SAM2_|SAT_|DRY_RUN|GPU_ID|EPOCHS|BATCH_SIZE|LR|SEED|GRAD_ACCUM_STEPS|CHECKPOINT_DIR)=' | while IFS='=' read -r key value; do
                if is_sensitive_env_key "${key}"; then
                    echo "  ${key}=***REDACTED***"
                else
                    echo "  ${key}=${value}"
                fi
            done
        fi

        echo "--------------------------------------------------"
    } | tee -a "${LOG_FILE}"
}

write_experiment_snapshot() {
    {
        echo "=================================================="
        echo "[实验配置快照] 单 GPU 集成测试 (单流卫星分支)"
        echo "timestamp: ${TIMESTAMP}"
        echo "host: $(hostname)"
        echo "project_root: ${PROJECT_ROOT}"
        echo "python_bin: ${PYTHON_BIN}"
        echo "gpu_id: ${GPU_ID}"
        echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
        echo "dry_run: ${DRY_RUN}"
        echo "log_terminal_env: ${LOG_TERMINAL_ENV}"
        echo "log_terminal_env_include_all: ${LOG_TERMINAL_ENV_INCLUDE_ALL}"
        echo "seed: ${SEED}"
        echo "epochs: ${EPOCHS}"
        echo "batch_size: ${BATCH_SIZE}"
        echo "grad_accum_steps: ${GRAD_ACCUM_STEPS}"
        echo "lr: ${LR}"
        echo "sam2_ckpt: ${SAM2_CKPT}"
        echo "sat_data_root: ${SAT_DATA_ROOT}"
        echo "config: rsprompter_anchor_satS_v11_sam2_large_full.py"
        echo "checkpoint_dir(default_if_unset): /data/wangcheng/checkpoint/test_iimr_integration_single_gpu_${TIMESTAMP}"
        echo "components.iimr:"
        echo "  $(format_switch iimr_enabled "${IIMR_ENABLED}" -1)"
        echo "  $(format_switch iimr_num_iterations "${IIMR_NUM_ITERATIONS}" -1)"
        echo "  $(format_switch iimr_memory_residual_weight "${IIMR_MEMORY_RESIDUAL_WEIGHT}" -1.0)"
        echo "  $(format_switch iimr_enable_after_epochs "${IIMR_ENABLE_AFTER_EPOCHS}" -1)"
        echo "  $(format_switch iimr_residual_warmup_epochs "${IIMR_RESIDUAL_WARMUP_EPOCHS}" -1)"
        echo "  $(format_switch iimr_residual_start_weight "${IIMR_RESIDUAL_START_WEIGHT}" -1.0)"
        echo "  $(format_switch iimr_use_topology_memory "${IIMR_USE_TOPOLOGY_MEMORY}" -1)"
        echo "  $(format_switch iimr_use_geometry_aware_kv "${IIMR_USE_GEOMETRY_AWARE_KV}" -1)"
        echo "  $(format_switch iimr_topology_to_decoder "${IIMR_TOPOLOGY_TO_DECODER}" -1)"
        echo "  $(format_switch iimr_use_gt_topology_tokens "${IIMR_USE_GT_TOPOLOGY_TOKENS}" -1)"
        echo "  $(format_switch iimr_gt_topology_only_first_iter "${IIMR_GT_TOPOLOGY_ONLY_FIRST_ITER}" -1)"
        echo "  $(format_switch iimr_use_dynamic_prompting "${IIMR_USE_DYNAMIC_PROMPTING}" -1)"
        echo "  $(format_switch iimr_detach_mask_path "${IIMR_DETACH_MASK_PATH}" -1)"
        echo "components.dynamic_prompting:"
        echo "  $(format_switch iimr_num_uncertainty_points "${IIMR_NUM_UNCERTAINTY_POINTS}" -1)"
        echo "  $(format_switch iimr_dynamic_balance_ratio "${IIMR_DYNAMIC_BALANCE_RATIO}" -1.0)"
        echo "  $(format_switch iimr_dynamic_min_point_distance "${IIMR_DYNAMIC_MIN_POINT_DISTANCE}" -1)"
        echo "  $(format_switch iimr_dynamic_start_iteration "${IIMR_DYNAMIC_START_ITERATION}" -1)"
        echo "  $(format_switch iimr_dynamic_point_mix "${IIMR_DYNAMIC_POINT_MIX}" -1.0)"
        if [ -n "${IIMR_DYNAMIC_FUSION_MODE}" ]; then
            echo "  iimr_dynamic_fusion_mode=${IIMR_DYNAMIC_FUSION_MODE}"
        else
            echo "  iimr_dynamic_fusion_mode=inherit(default)"
        fi
        echo "=================================================="
    } | tee -a "${LOG_FILE}"
}

: > "${LOG_FILE}"
write_experiment_snapshot
write_terminal_env_snapshot

echo "[INFO] 日志目录: ${LOG_DIR}"
echo "[INFO] 日志文件: ${LOG_FILE}"

log_test_result "${LOG_FILE}" "环境初始化" "PASS" "GPU ${GPU_ID}, conda cvt2"

echo ""
echo "[INFO] 开始 IIMR 组件导入测试..."
"${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

print('测试 IIMR 组件导入...')

from rsprompter.iterative_memory import SimplifiedMemoryBank, UncertaintyPointGenerator, IterativeMemoryReasoner
print('✓ iterative_memory 模块导入成功')

from rsprompter.topology_tokenizer import TopologyTokenizer, BoundaryExtractor
print('✓ topology_tokenizer 模块导入成功')

from models.losses import SoftClDiceLoss
print('✓ losses 模块导入成功')

print('所有 IIMR 组件导入成功!')
" 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_test_result "${LOG_FILE}" "IIMR组件导入" "PASS"
else
    log_test_result "${LOG_FILE}" "IIMR组件导入" "FAIL" "导入失败"
    exit 1
fi

echo ""
echo "[INFO] 开始模型构建测试..."
"${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

import torch
print('测试模型构建...')

from mmengine.config import Config
cfg = Config.fromfile('${PROJECT_ROOT}/configs/rsprompter_anchor_satS_v11_sam2_large_full.py')
print('✓ 配置文件加载成功')

print('模型配置:')
print(f'  model.type: {cfg.model.type}')
print(f'  enable_drone_branch: {cfg.model.get(\"enable_drone_branch\", \"N/A\")}')

print('模型构建测试完成!')
" 2>&1 | tee -a "${LOG_FILE}"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_test_result "${LOG_FILE}" "模型配置加载" "PASS"
else
    log_test_result "${LOG_FILE}" "模型配置加载" "FAIL"
    exit 1
fi


if [ "${DRY_RUN}" = "1" ]; then
    echo "[INFO] DRY_RUN 模式，跳过训练测试"
    log_test_result "${LOG_FILE}" "训练测试" "SKIP" "DRY_RUN 模式"
else
    echo ""
    echo "[INFO] 开始训练流程测试 (${EPOCHS} epoch)..."

    CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/wangcheng/checkpoint/test_iimr_integration_single_gpu_${TIMESTAMP}}"
    mkdir -p "${CHECKPOINT_DIR}"

    TRAIN_BIN="${PROJECT_ROOT}/train/train_rsprompter_fusion.py"
    CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_anchor_satS_v11_sam2_large_full.py"

    if [ ! -f "${TRAIN_BIN}" ]; then
        echo "[WARN] 训练脚本不存在: ${TRAIN_BIN}"
        log_test_result "${LOG_FILE}" "训练测试" "SKIP" "训练脚本不存在"
    elif [ ! -d "${SAT_DATA_ROOT}" ]; then
        echo "[WARN] 数据集不存在: ${SAT_DATA_ROOT}"
        log_test_result "${LOG_FILE}" "训练测试" "SKIP" "数据集不存在"
    else
        # 构建训练命令
        CMD=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node 1
            "${TRAIN_BIN}"
            --config "${CONFIG_PATH}"
            --data-root "${SAT_DATA_ROOT}"
            --image-size 512 512
            --batch-size "${BATCH_SIZE}"
            --epochs "${EPOCHS}"
            --lr "${LR}"
            --seed "${SEED}"
            --checkpoint-dir "${CHECKPOINT_DIR}"
            --val-ratio 0.2
            --val-batch-size "${BATCH_SIZE}"
            --val-every-n-epochs 1
            --early-stopping-patience 10
            --sat-backbone-lr-mult 1.0
            --sat-other-lr-mult 1.0
            --grad-accum-steps "${GRAD_ACCUM_STEPS}"
            --warmup-iters 50
            --weight-decay 0.05
            --deterministic 1
            --amp 0)

        # 添加 IIMR 参数（仅在设置时传递, -1 表示继承 config 默认值）
        [ "${IIMR_ENABLED}" -ge 0 ] && CMD+=(--iimr-enabled "${IIMR_ENABLED}")                           # 总开关
        [ "${IIMR_NUM_ITERATIONS}" -gt 0 ] && CMD+=(--iimr-num-iterations "${IIMR_NUM_ITERATIONS}")       # 迭代次数
        [ "${IIMR_MEMORY_RESIDUAL_WEIGHT}" != "-1.0" ] && CMD+=(--iimr-memory-residual-weight "${IIMR_MEMORY_RESIDUAL_WEIGHT}")  # 残差融合系数
        [ "${IIMR_ENABLE_AFTER_EPOCHS}" -ge 0 ] && CMD+=(--iimr-enable-after-epochs "${IIMR_ENABLE_AFTER_EPOCHS}")  # 前N个epoch关闭IIMR
        [ "${IIMR_RESIDUAL_WARMUP_EPOCHS}" -ge 0 ] && CMD+=(--iimr-residual-warmup-epochs "${IIMR_RESIDUAL_WARMUP_EPOCHS}")  # residual warmup epoch数
        [ "${IIMR_RESIDUAL_START_WEIGHT}" != "-1.0" ] && CMD+=(--iimr-residual-start-weight "${IIMR_RESIDUAL_START_WEIGHT}")  # residual warmup起点
        [ "${IIMR_USE_ROI_POS_ENCODING}" -ge 0 ] && CMD+=(--iimr-use-roi-pos-encoding "${IIMR_USE_ROI_POS_ENCODING}")            # ROI空间位置编码
        [ "${IIMR_USE_TOPOLOGY_MEMORY}" -ge 0 ] && CMD+=(--iimr-use-topology-memory "${IIMR_USE_TOPOLOGY_MEMORY}")              # 拓扑token注入KV
        [ "${IIMR_USE_GEOMETRY_AWARE_KV}" -ge 0 ] && CMD+=(--iimr-use-geometry-aware-kv "${IIMR_USE_GEOMETRY_AWARE_KV}")       # Geometry-aware KV
        [ "${IIMR_TOPOLOGY_TO_DECODER}" -ge 0 ] && CMD+=(--iimr-topology-to-decoder "${IIMR_TOPOLOGY_TO_DECODER}")               # 拓扑送decoder
        [ "${IIMR_USE_GT_TOPOLOGY_TOKENS}" -ge 0 ] && CMD+=(--iimr-use-gt-topology-tokens "${IIMR_USE_GT_TOPOLOGY_TOKENS}")      # GT拓扑token
        [ "${IIMR_GT_TOPOLOGY_ONLY_FIRST_ITER}" -ge 0 ] && CMD+=(--iimr-gt-topology-only-first-iter "${IIMR_GT_TOPOLOGY_ONLY_FIRST_ITER}") # GT仅首次
        [ "${IIMR_USE_DYNAMIC_PROMPTING}" -ge 0 ] && CMD+=(--iimr-use-dynamic-prompting "${IIMR_USE_DYNAMIC_PROMPTING}")         # 动态提示开关
        [ "${IIMR_DETACH_MASK_PATH}" -ge 0 ] && CMD+=(--iimr-detach-mask-path "${IIMR_DETACH_MASK_PATH}")                        # 梯度隔离
        [ "${IIMR_NUM_UNCERTAINTY_POINTS}" -gt 0 ] && CMD+=(--iimr-num-uncertainty-points "${IIMR_NUM_UNCERTAINTY_POINTS}")       # 采样点数
        [ "${IIMR_DYNAMIC_BALANCE_RATIO}" != "-1.0" ] && CMD+=(--iimr-dynamic-balance-ratio "${IIMR_DYNAMIC_BALANCE_RATIO}")     # 正负比例
        [ "${IIMR_DYNAMIC_MIN_POINT_DISTANCE}" -ge 0 ] && CMD+=(--iimr-dynamic-min-point-distance "${IIMR_DYNAMIC_MIN_POINT_DISTANCE}") # 最小距离
        [ "${IIMR_DYNAMIC_START_ITERATION}" -gt 0 ] && CMD+=(--iimr-dynamic-start-iteration "${IIMR_DYNAMIC_START_ITERATION}")    # 起始迭代
        [ "${IIMR_DYNAMIC_POINT_MIX}" != "-1.0" ] && CMD+=(--iimr-dynamic-point-mix "${IIMR_DYNAMIC_POINT_MIX}")                 # 嵌入强度
        [ -n "${IIMR_DYNAMIC_FUSION_MODE}" ] && CMD+=(--iimr-dynamic-fusion-mode "${IIMR_DYNAMIC_FUSION_MODE}")                   # 融合模式

        {
            echo "[CMD] 训练命令快照:"
            printf '  %q ' "${CMD[@]}"
            echo
            echo "--------------------------------------------------"
        } | tee -a "${LOG_FILE}"

        # 执行训练命令
        "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            log_test_result "${LOG_FILE}" "训练测试" "PASS" "${EPOCHS} epoch 完成"
        else
            log_test_result "${LOG_FILE}" "训练测试" "FAIL"
        fi
    fi
fi

echo ""
echo "[INFO] 记录 GPU 内存状态..."
get_gpu_memory "${GPU_ID}" | tee -a "${LOG_FILE}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_STR="${DURATION} 秒"

if [ ${DURATION} -gt 60 ]; then
    DURATION_STR="$((DURATION / 60)) 分 $((DURATION % 60)) 秒"
fi

print_test_footer "单 GPU 集成测试 (单流卫星分支)" "完成" "${DURATION_STR}"

echo ""
echo "日志文件: ${LOG_FILE}"
echo "检查点目录: ${CHECKPOINT_DIR:-N/A}"
