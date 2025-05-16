#!/usr/bin/env bash
set -euo pipefail

cp ../bench.sh .
chmod a+x ./bench.sh

./bench.sh p2pBandwidthLatencyTest 2 /home/xtang/p2pBandwidthLatencyTest/ " " --device=0,1 --mode=range --size=4KiB,256MiB --iters=1000

rm -rf ./bench.sh