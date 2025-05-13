#!/usr/bin/env bash
#
# usage: ./run_nccl_tests.sh <num_ranks> [Test1 Test2 ...]
#
# Available tests: AllReduce, ReduceScatter, AllGather, Broadcast, Reduce, AlltoAll

set -euo pipefail

usage() {
  echo "Usage: $0 <num_ranks> [Test1 Test2 ...]"
  echo "Available tests: AllReduce, ReduceScatter, AllGather, Broadcast, Reduce, AlltoAll"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

# 1. 拿到 GPU 数
n="$1"; shift

# 2. 如果没有后续参数，则跑全部
if [ $# -gt 0 ]; then
  tests=("$@")
else
  tests=(AllReduce ReduceScatter AllGather Broadcast Reduce AlltoAll)
fi

# 3. 映射：测试名 -> 可执行二进制
declare -A bins=(
  [AllReduce]=build/all_reduce_perf
  [ReduceScatter]=build/reduce_scatter_perf
  [AllGather]=build/all_gather_perf
  [Broadcast]=build/broadcast_perf
  [Reduce]=build/reduce_perf
  [AlltoAll]=build/alltoall_perf
)

# 4. 动态计算归一化因子
#    AllReduce 的 factor = 2*(n-1)/n ，其余按需求修改
allreduce_factor=$(printf "%.2f" "$(echo "scale=6; 2*($n-1)/$n" | bc -l)")
reducescatter_factor=$(printf "%.2f" "$(echo "scale=6; ($n-1)/$n" | bc -l)")
allgather_factor=$reducescatter_factor
broadcast_factor="1.00"
reduce_factor="1.00"
alltoall_factor=$reducescatter_factor

declare -A factors=(
  [AllReduce]="$allreduce_factor"
  [ReduceScatter]="$reducescatter_factor"
  [AllGather]="$allgather_factor"
  [Broadcast]="$broadcast_factor"
  [Reduce]="$reduce_factor"
  [AlltoAll]="$alltoall_factor"
)

# 5. 消息大小范围
START=128000000
END=1024000000

# 6. 输出 CSV 文件
OUTFILE="nccl_results_rank${n}.csv"
echo "Test,Size(Bytes),Count(Elements),Type,OutPlaceTime(us),OutPlaceAlgBW(GB/s),OutPlaceBusBW(GB/s),InPlaceTime(us),InPlaceAlgBW(GB/s),InPlaceBusBW(GB/s)" > "$OUTFILE"

# 7. 运行并解析函数
run_test() {
  local binpath="$1"
  local label="$2"
  local factor="$3"

  if [ ! -x "$binpath" ]; then
    echo "Error: binary '$binpath' not found or not executable"
    exit 1
  fi

  echo -e "\n=== Running $label (factor=$factor) ==="
  "$binpath" -b "$START" -e "$END" -f "$factor" -g "$n" \
    | tee "raw_${label}.log" \
    | awk -v test="$label" '
        /^[[:space:]]*[0-9]+/ {
          printf "%s,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", \
            test, $1, $2, $3, $6, $7, $8, $10, $11, $12
        }' \
    >> "$OUTFILE"
}

export CUDA_VISIBLE_DEVICES=0

# 8. 按用户指定的列表依次跑
for label in "${tests[@]}"; do
  if [[ -v bins[$label] ]]; then
    run_test "${bins[$label]}" "$label" "${factors[$label]}"
  else
    echo "Warning: unknown test '$label', skipping"
  fi
done

echo -e "\nAll done. Results saved in '$OUTFILE'."
