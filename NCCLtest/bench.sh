#!/usr/bin/env bash
## 通用 benchmark 运行脚本：按 GPU 循环做 CPU 亲和，并根据表头自动定位 CPU/NUMA Affinity 列
set -euo pipefail

usage() {
  echo "Usage: $0 <BenchName> <GPU_LIST> <BenchDir> <BenchCmd> [<bench_cmd_args>...]"
  echo "  BenchName       : 用于结果目录和文件前缀的测试名"
  echo "  GPU_LIST        : 要使用的 GPU ID 列表，用逗号分隔（例如 0,1,2,3）"
  echo "  BenchDir        : 测试所在目录（脚本会 cd 到这里）"
  echo "  BenchCmd        : 在 BenchDir 下可执行的测试文件或脚本"
  echo "  bench_cmd_args  : 传给 BenchCmd 的额外参数（可选）"
  exit 1
}

if [[ $# -lt 4 ]]; then
  usage
fi

BenchName="$1"
GPU_LIST="$2"            # e.g. "0,1,2,3"
BenchDir="$3"
BenchCmd="$4"
shift 4
BenchArgs=("$@")

# 解析 GPU 列表
IFS=',' read -ra GPUS <<< "$GPU_LIST"
GPU_NUM=${#GPUS[@]}
echo "GPU_NUM: ${GPU_NUM}"

# 只让测试程序看到这几张卡
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

# 准备结果目录
WKDIR=$(pwd)
RESULT_DIR="${WKDIR}/${BenchName}/GPUs_${GPU_LIST}"
mkdir -p "${RESULT_DIR}"

# 进入测试目录
cd "${BenchDir}"

echo
echo "=== 使用 GPUs: ${GPU_LIST} (共 ${GPU_NUM} 张) ==="
echo "=== GPU 拓扑 ==="
nvidia-smi topo -m
echo
echo "=== GPU 状态 ==="
nvidia-smi
echo

# 根据表头自动定位 “CPU Affinity” 和 “NUMA Affinity” 列号
read -r header_line <<< "$(nvidia-smi topo -m | head -n1)"
# 分词
read -ra header <<< "${header_line}"
for i in "${!header[@]}"; do
  if [[ "${header[$i]}" == "CPU" && "${header[$i+1]}" == "Affinity" ]]; then
    cpu_col=$((i+2))    # awk 字段从 1 开始，i 从 0 开始
  fi
  if [[ "${header[$i]}" == "NUMA" && "${header[$i+1]}" == "Affinity" ]]; then
    numa_col=$((i+2))
  fi
done

# 为每张 GPU 计算两颗亲和 CPU 核心
declare -A GPU_CPU0 GPU_CPU1
for idx in "${!GPUS[@]}"; do
  # 提取对应行的 CPU 和 NUMA Affinity
  cpu_list=$(nvidia-smi topo -m | awk -v c="$cpu_col" -v g="GPU${idx}" '$1==g {print $c}')
  numa_affinity=$(nvidia-smi topo -m | awk -v n="$numa_col" -v g="GPU${idx}" '$1==g {print $n}')
  echo "GPU${GPUS[idx]} 的 CPU Affinity: ${cpu_list}, NUMA Affinity: ${numa_affinity}"

  # 拆分 CPU Affinity 为独立核心列表
  IFS=',' read -ra ranges <<< "${cpu_list}"
  cores=()
  for r in "${ranges[@]}"; do
    if [[ "${r}" == *-* ]]; then
      IFS='-' read -r s e <<< "${r}"
      cores+=($(seq "${s}" "${e}"))
    else
      cores+=("${r}")
    fi
  done

  GPU_CPU0[$idx]=${cores[0]}
  GPU_CPU1[$idx]=${cores[1]:-${cores[0]}}
done

# 取第一张 GPU 对应的 CPU0 读频率范围（全系统共享）
CPU0_first=${GPU_CPU0[0]}
maxfreq=$(< "/sys/bus/cpu/devices/cpu${CPU0_first}/cpufreq/cpuinfo_max_freq")
minfreq=$(< "/sys/bus/cpu/devices/cpu${CPU0_first}/cpufreq/cpuinfo_min_freq")

# 定义：对某张 GPU 执行一次测试并保存结果
run_test_for_gpu() {
  local idx=$1     # 数组索引，从 0 开始
  local mode=$2
  local label=$3
  local c0=${GPU_CPU0[$idx]}
  local c1=${GPU_CPU1[$idx]}
  echo ">>> [${mode}] GPU${GPUS[idx]} 绑定 CPU ${c0},${c1} 运行 ${BenchCmd}"
  turbostat -c "${c0},${c1}" -i2 -n4 -q &
  taskset -c "${c0}" ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}"

  mv nccl_results_rank${GPU_NUM}.csv \
     "${RESULT_DIR}/${BenchName}_GPU${GPUS[idx]}_${label}.csv"
}

##############################################################################
## 1) powersave & performance 模式
##############################################################################
for gov in powersave performance; do
  echo; echo "=== 设置 CPU governor 为 ${gov} ==="
  cpupower frequency-set -g "${gov}" >/dev/null 2>&1

  for idx in "${!GPUS[@]}"; do
    run_test_for_gpu "$idx" "$gov" "${gov}"
  done
done

##############################################################################
## 2) 频率扫描
##############################################################################
echo; echo "=== 频率扫描 from ${minfreq} to ${maxfreq} step 200000 ==="
# 检测 userspace governor 是否可用
avail=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors)
if [[ " $avail " == *" userspace "* ]]; then
  echo "=== 切换到 userspace governor ==="
  cpupower frequency-set -g userspace
else
  echo "!!! 警告：当前不支持 userspace gover"
fi


for freq in $(seq "${minfreq}" 200000 "${maxfreq}"); do
  echo; echo ">>> 设定 CPU 频率到 ${freq}"
  cpupower frequency-set -d "${freq}" -u "${freq}" >/dev/null 2>&1
  for idx in "${!GPUS[@]}"; do
    run_test_for_gpu "$idx" "freq${freq}" "freq${freq}"
  done
done

# 恢复默认频率和 governor
cpupower frequency-set -d "${minfreq}" -u "${maxfreq}" >/dev/null 2>&1
cpupower frequency-set -g performance >/dev/null 2>&1

##############################################################################
## 3) （可选）收集 emon 数据 —— 仅 Intel 平台
##############################################################################
#if lscpu | grep -q GenuineIntel; then
if lscpu | grep -q GenuineIntel && [[ -f /opt/intel/sep/sep_vars.sh ]]; then
  echo; echo "=== 收集 emon 数据 ==="
  export CPATH="${CPATH:-}"
  export LIBRARY_PATH="${LIBRARY_PATH:-}"
  export PYTHONPATH="${PYTHONPATH:-}"

  echo; echo "=== 收集 emon 数据 ==="
  # 保存外层脚本的位置参数
  _saved_args=("$@")
  # 临时关闭 nounset 检查
  set +u
  # 给 sep_vars.sh 一个“dummy”参数
  set -- dummy
  # 再 source，脚本里的 $1 就有值了
  . /opt/intel/sep/sep_vars.sh
  # 恢复 nounset
  set -u
  # 恢复外层脚本的位置参数
  set -- "${_saved_args[@]}"

  #. /opt/intel/sep/sep_vars.sh
  for idx in "${!GPUS[@]}"; do
    c0=${GPU_CPU0[$idx]}
    emon -collect-edp -f "${RESULT_DIR}/emon-${BenchName}-GPU${GPUS[idx]}.dat" &
    taskset -c "${c0}" ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}"
    emon -stop
  done
fi

##############################################################################
## 4) 收集 NSYS 数据
##############################################################################
echo; echo "=== 收集 NSYS 数据 ==="
for freq in "${minfreq}" "${maxfreq}"; do
  cpupower frequency-set -d "${freq}" -u "${freq}" >/dev/null 2>&1
  for idx in "${!GPUS[@]}"; do
    c0=${GPU_CPU0[$idx]}
    /usr/local/cuda/bin/nsys profile \
      --trace=cuda,nvtx,osrt --sample=none --force-overwrite true \
      -o "${RESULT_DIR}/nsys-${BenchName}-GPU${GPUS[idx]}-freq${freq}" --stats=true \
      taskset -c "${c0}" ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}"
  done
done

cd "${WKDIR}"
echo; echo "All done. 结果保存在 ${RESULT_DIR}"

