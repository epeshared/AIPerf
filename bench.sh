#!/usr/bin/env bash
## 通用 benchmark 运行脚本：可以在任意测试目录下调用可执行文件
set -euo pipefail

usage() {
  echo "Usage: $0 <BenchName> <GPU_NUM> <BenchDir> <BenchCmd> [<bench_cmd_args>...]"
  echo "  BenchName       : 用于结果目录和文件前缀的测试名"
  echo "  GPU_NUM         : 要使用的 GPU 数量"
  echo "  BenchDir        : 测试所在目录（脚本会 cd 到这里）"
  echo "  BenchCmd        : 在 BenchDir 下可执行的测试文件或脚本"
  echo "  bench_cmd_args  : 传给 BenchCmd 的额外参数（可选）"
  exit 1
}

if [[ $# -lt 4 ]]; then
  usage
fi

BenchName="$1"
GPU_NUM="$2"
BenchDir="$3"
BenchCmd="$4"
shift 4
BenchArgs=("$@")

# 准备结果目录
WKDIR=$(pwd)
RESULT_DIR="${WKDIR}/${BenchName}/GPU_NUM_${GPU_NUM}"
mkdir -p "${RESULT_DIR}"

# 进入测试目录
cd "${BenchDir}"

echo
echo "=== GPU 拓扑 ==="
nvidia-smi topo -m
echo
echo "=== GPU 状态 ==="
nvidia-smi
echo

# 找出和 GPU0 关联的 NUMA 节点和 CPU 列表
GPU0_NUMA_AFFINITY=$(nvidia-smi topo -m | awk '/^GPU0/ {print $4}')
GPU0_CPU_LIST=$(nvidia-smi topo -m | awk '/^GPU0/ {print $3}')

# 拆分成单个 core 数组
IFS=',' read -ra ranges <<< "${GPU0_CPU_LIST}"
cpu_cores=()
for r in "${ranges[@]}"; do
  if [[ "${r}" == *-* ]]; then
    IFS='-' read -r s e <<< "${r}"
    cpu_cores+=($(seq "${s}" "${e}"))
  else
    cpu_cores+=("${r}")
  fi
done

# 取前两个 core 用于绑定
CPU0=${cpu_cores[0]}
CPU1=${cpu_cores[1]:-${cpu_cores[0]}}

# 判断最高/最低频率
maxfreq=$(< /sys/bus/cpu/devices/cpu${CPU0}/cpufreq/cpuinfo_max_freq)
minfreq=$(< /sys/bus/cpu/devices/cpu${CPU0}/cpufreq/cpuinfo_min_freq)

# Helper: 执行一次测试并保存结果
run_test() {
  local mode=$1   # powersave/performance or freq value
  local label=$2  # 文件后缀标识
  echo ">>> [$mode] 绑定 CPU ${core} 运行 ${BenchCmd} ${BenchArgs[*]}"
  turbostat -c ${CPU0},${CPU1} -i2 -n4 -q &
  taskset -c ${core} ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}"
  mv nccl_results_rank${GPU_NUM}.csv \
     "${RESULT_DIR}/${BenchName}_GPU${GPU_NUM}_${label}.csv"
}

##############################################################################
## 1) powersave & performance 模式
##############################################################################
for gov in powersave performance; do
  echo
  echo "=== 设置 CPU governor 为 ${gov} ==="
  cpupower frequency-set -g "${gov}" >/dev/null 2>&1

  for core in "${CPU0}" "${CPU1}"; do
    run_test "${gov}" "${gov}_CPU${core}"
  done
done

##############################################################################
## 2) 频率扫描
##############################################################################
echo
echo "=== 频率扫描 from ${minfreq} to ${maxfreq} step 200000 ==="
cpupower frequency-set -g userspace >/dev/null 2>&1

for freq in $(seq "${minfreq}" 200000 "${maxfreq}"); do
  echo ">>> 设定 CPU 频率到 ${freq}"
  cpupower frequency-set -d "${freq}" -u "${freq}" >/dev/null 2>&1

  core=${CPU0}
  turbostat -c ${CPU0},${CPU1} -i2 -n4 -q &
#   taskset -c ${core} ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}"

  # 把屏幕输出重定向到 .txt 文件
  LOGFILE="${RESULT_DIR}/${BenchName}_GPU${GPU_NUM}_freq${freq}.txt"
  echo "Running ${BenchCmd}, logging to ${LOGFILE}"
  taskset -c ${core} ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}" \
    > "${LOGFILE}" 2>&1

  mv nccl_results_rank${GPU_NUM}.csv \
     "${RESULT_DIR}/${BenchName}_GPU${GPU_NUM}_freq${freq}.csv"
done

# 恢复默认频率
cpupower frequency-set -d "${minfreq}" -u "${maxfreq}" >/dev/null 2>&1
cpupower frequency-set -g performance >/dev/null 2>&1

##############################################################################
## 3) （可选）收集 emon / nsys 数据 —— 仅 Intel 平台
##############################################################################
if lscpu | grep -q GenuineIntel; then
  echo; echo "=== 收集 emon 数据 ==="
  . /opt/intel/sep/sep_vars.sh
  core=${CPU0}
  emon -collect-edp -f "emon-${BenchName}.dat" &
  taskset -c ${core} ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}"
  emon -stop
fi


echo; echo "=== 收集 NSYS 数据 ==="
for freq in "${minfreq}" "${maxfreq}"; do
cpupower frequency-set -d "${freq}" -u "${freq}" >/dev/null 2>&1
/usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx,osrt --sample=none --force-overwrite true \
    -o "${RESULT_DIR}/nsys-${BenchName}-freq${freq}" --stats=true \
    taskset -c ${CPU0} ./"${BenchCmd}" "${GPU_NUM}" "${BenchName}" "${BenchArgs[@]}"
done

cd "${WKDIR}"
echo
echo "All done. 结果保存在 ${RESULT_DIR}"