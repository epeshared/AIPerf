#!/usr/bin/env bash
set -euo pipefail

#cp ../bench.sh .
chmod a+x ./bench.sh

./bench.sh AllReduce     "0,1" /home/xtang/nccl-tests/ nccl_test.sh
#./bench.sh ReduceScatter 2 /home/xtang/nccl-tests/ nccl_test.sh
#./bench.sh AllGather     2 /home/xtang/nccl-tests/ nccl_test.sh
#./bench.sh Broadcast     2 /home/xtang/nccl-tests/ nccl_test.sh
#./bench.sh Reduce        2 /home/xtang/nccl-tests/ nccl_test.sh
#./bench.sh AlltoAll      2 /home/xtang/nccl-tests/ nccl_test.sh

#rm -rf ./bench.sh
