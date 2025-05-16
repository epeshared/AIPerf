#!/usr/bin/env bash
set -euo pipefail

cp ../bench.sh .
chmod a+x ./bench.sh

./bench.sh AllReduce     1 /home/xtang/nccl-tests/ nccl_test.sh
./bench.sh ReduceScatter 1 /home/xtang/nccl-tests/ nccl_test.sh
./bench.sh AllGather     1 /home/xtang/nccl-tests/ nccl_test.sh
./bench.sh Broadcast     1 /home/xtang/nccl-tests/ nccl_test.sh
./bench.sh Reduce        1 /home/xtang/nccl-tests/ nccl_test.sh
./bench.sh AlltoAll      1 /home/xtang/nccl-tests/ nccl_test.sh

rm -rf ./bench.sh