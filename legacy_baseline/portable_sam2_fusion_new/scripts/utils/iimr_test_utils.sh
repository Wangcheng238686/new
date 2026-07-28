#!/bin/bash
# IIMR 集成测试工具函数
# 提供环境激活、日志管理、测试结果记录等公共功能

set -e

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cvt2}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/wangcheng/envs/cvt2}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${PROJECT_ROOT}/logs/iimr_test}"

activate_env() {
    echo "[INFO] 激活 conda 环境: ${CONDA_ENV_NAME}"
    
    # 将 cvt2 环境的 bin 目录添加到 PATH 最前面
    if [ -d "${CONDA_ENV_PATH}/bin" ]; then
        export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
        echo "[INFO] 已将 ${CONDA_ENV_PATH}/bin 添加到 PATH"
    fi
    
    # 直接使用环境路径中的 Python
    if [ -f "${CONDA_ENV_PATH}/bin/python" ]; then
        PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
        echo "[INFO] 使用环境 Python: ${PYTHON_BIN}"
    elif [ -f "/data/wangcheng/envs/cvt2/bin/python" ]; then
        export PATH="/data/wangcheng/envs/cvt2/bin:${PATH}"
        PYTHON_BIN="/data/wangcheng/envs/cvt2/bin/python"
        echo "[INFO] 使用环境 Python: ${PYTHON_BIN}"
    else
        echo "[WARN] 未找到环境 Python，尝试使用系统 Python"
        PYTHON_BIN=$(which python)
    fi
    
    # 验证 Python 可用性
    if [ ! -f "${PYTHON_BIN}" ]; then
        echo "[ERROR] 找不到 Python 解释器"
        return 1
    fi
    
    echo "[INFO] Python 版本: $(${PYTHON_BIN} --version 2>&1)"
    echo "[INFO] CUDA 可用: $(${PYTHON_BIN} -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo "无法检测")"
    
    export PYTHON_BIN
    return 0
}

setup_log_dir() {
    local log_type="${1:-single_gpu}"
    local log_dir="${LOG_BASE_DIR}/${log_type}"
    
    mkdir -p "${log_dir}"
    
    # 只返回目录路径，不输出其他信息
    echo "${log_dir}"
}

get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

log_test_result() {
    local log_file="${1}"
    local test_name="${2}"
    local status="${3}"
    local details="${4:-}"
    
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    {
        echo "========================================"
        echo "[${timestamp}] 测试: ${test_name}"
        echo "状态: ${status}"
        if [ -n "${details}" ]; then
            echo "详情: ${details}"
        fi
        echo "========================================"
    } >> "${log_file}"
    
    if [ "${status}" = "PASS" ]; then
        echo "[PASS] ${test_name}"
    else
        echo "[FAIL] ${test_name}: ${details}"
    fi
}

check_gpu_available() {
    local gpu_id="${1:-0}"
    local num_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
    
    if [ "${num_gpus}" -lt 1 ]; then
        echo "[ERROR] 没有检测到 GPU"
        return 1
    fi
    
    if [ "${gpu_id}" -ge "${num_gpus}" ]; then
        echo "[ERROR] GPU ${gpu_id} 不存在，可用 GPU 数量: ${num_gpus}"
        return 1
    fi
    
    echo "[INFO] 检测到 ${num_gpus} 张 GPU"
    nvidia-smi -L
    return 0
}

get_gpu_memory() {
    local gpu_id="${1:-0}"
    nvidia-smi -i "${gpu_id}" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "已用: %dMB / 总计: %dMB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
}

run_python_test() {
    local python_bin="${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}"
    local test_script="${1}"
    local log_file="${2:-/dev/null}"
    shift 2
    local extra_args="$@"
    
    echo "[INFO] 运行测试: ${test_script}"
    echo "[INFO] 参数: ${extra_args}"
    
    if [ "${log_file}" = "/dev/null" ]; then
        "${python_bin}" "${test_script}" ${extra_args}
    else
        "${python_bin}" "${test_script}" ${extra_args} 2>&1 | tee -a "${log_file}"
    fi
    
    return ${PIPESTATUS[0]}
}

print_test_header() {
    local test_name="${1}"
    local config_info="${2:-}"
    
    echo ""
    echo "========================================"
    echo "IIMR 集成测试: ${test_name}"
    echo "========================================"
    echo "时间: $(date)"
    echo "项目目录: ${PROJECT_ROOT}"
    if [ -n "${config_info}" ]; then
        echo "${config_info}"
    fi
    echo "========================================"
}

print_test_footer() {
    local test_name="${1}"
    local status="${2}"
    local duration="${3:-}"
    
    echo ""
    echo "========================================"
    echo "测试完成: ${test_name}"
    echo "状态: ${status}"
    if [ -n "${duration}" ]; then
        echo "耗时: ${duration}"
    fi
    echo "========================================"
}

generate_comparison_report() {
    local baseline_log="${1}"
    local iimr_log="${2}"
    local output_file="${3}"
    
    echo "[INFO] 生成对比报告: ${output_file}"
    
    {
        echo "========================================"
        echo "IIMR 消融实验对比报告"
        echo "生成时间: $(date)"
        echo "========================================"
        echo ""
        echo "--- Baseline (IIMR 禁用) ---"
        echo "日志文件: ${baseline_log}"
        if [ -f "${baseline_log}" ]; then
            grep -E "(loss|mIoU|Acc|GPU|内存|耗时)" "${baseline_log}" | tail -20
        fi
        echo ""
        echo "--- IIMR 启用 ---"
        echo "日志文件: ${iimr_log}"
        if [ -f "${iimr_log}" ]; then
            grep -E "(loss|mIoU|Acc|GPU|内存|耗时)" "${iimr_log}" | tail -20
        fi
        echo ""
        echo "========================================"
    } > "${output_file}"
    
    echo "[INFO] 对比报告已保存"
}
