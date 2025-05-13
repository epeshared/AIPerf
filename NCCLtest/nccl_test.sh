#!/usr/bin/env bash
#
# usage: ./run_nccl_tests.sh <num_ranks>
#
# Runs all NCCL performance tests and logs output to both screen and CSV file for Excel.

if [ -z "$1" ]; then
  echo "Usage: $0 <num_ranks>"
  exit 1
fi

n="$1"
START=128000000
END=1024000000

# Calculate correction factors
allreduce_factor=$(printf "%.2f" "$(echo "scale=6; 2*(\$n-1)/\$n" | bc -l)")
reducescatter_factor=$(printf "%.2f" "$(echo "scale=6; (\$n-1)/\$n" | bc -l)")
allgather_factor=$reducescatter_factor
broadcast_factor="1.00"
reduce_factor="1.00"
alltoall_factor=$reducescatter_factor

# CSV output file
OUTFILE="nccl_results_rank${n}.csv"
echo "Test,Size(Bytes),Count(Elements),Type,OutPlaceTime(us),OutPlaceAlgBW(GB/s),OutPlaceBusBW(GB/s),InPlaceTime(us),InPlaceAlgBW(GB/s),InPlaceBusBW(GB/s)" > $OUTFILE

echo "Running NCCL tests with $n GPU(s), outputting to $OUTFILE"

# Helper to run a test and append parsed CSV
run_test() {
  local bin="$1"    # binary path
  local label="$2"  # CSV Test field
  local factor="$3"

  echo -e "\n=== $label ==="
  # Run test, tee output, parse numerical lines
  $bin -b $START -e $END -f $factor -g $n | tee raw_${label}.log | \
    awk -v test="$label" ' \
      /^[[:space:]]*[0-9]+/ { \
        # columns: $1=size, $2=count, $3=type, $6=time, $7=algbw, $8=busbw, \
        # $10=in-place time, $11=in-place algbw, $12=in-place busbw \
        printf "%s,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", \
          test, $1, $2, $3, $6, $7, $8, $10, $11, $12; \
      }' >> $OUTFILE
}

export CUDA_VISIBLE_DEVICES=0
factor=2
# Execute all tests
echo "Starting AllReduce"
run_test ./build/all_reduce_perf      AllReduce      $factor

echo "Starting ReduceScatter"
run_test ./build/reduce_scatter_perf  ReduceScatter  $factor

echo "Starting AllGather"
run_test ./build/all_gather_perf      AllGather      $factor

echo "Starting Broadcast"
run_test ./build/broadcast_perf       Broadcast      $factor

echo "Starting Reduce"
run_test ./build/reduce_perf          Reduce         $factor

echo "Starting AlltoAll"
run_test ./build/alltoall_perf        AlltoAll       $factor

echo -e "\nAll tests complete. Parsed results in $OUTFILE"
