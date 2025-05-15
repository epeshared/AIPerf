#!/usr/bin/bash
## superbenchmark kernel-launch micro-kernel benchmark

echo "CONDAROOT=$CONDAROOT"
echo "CONDAENV=$CONDAENV"

if [[ $CONDAROOT != "" ]]; then
    source $CONDAROOT/etc/profile.d/conda.sh
    conda activate $CONDAENV
    conda install -y zip unzip cmake gcc==12.2.0 gxx=12.2.0 bc
    apt install -y msr-tools
fi
echo "cmake is $(which cmake)"

echo ""
echo "GPU topology"

nvidia-smi topo -m
echo ""
echo "GPU smi"
nvidia-smi
echo ""

GPU0_NUMA_AFFINITY=`nvidia-smi topo -m | awk '/^GPU0/ {print $4}'`
GPU0_AFFINITY_CPULIST=`nvidia-smi topo -m | awk '/^GPU0/ {print $3}'`
CPUVENDER=$(lscpu |awk '/Vendor ID/ {print $3}')

IFS=',' read -ra ranges <<< "$GPU0_AFFINITY_CPULIST"
cpu_array=()

for range in "${ranges[@]}"; do
    if [[ "$range" == *-* ]]; then
        start=${range%-*}
        end=${range#*-}
        cpu_array+=($(seq "$start" "$end"))
    else
        cpu_array+=("$range")
    fi
done

# frequency scaling
maxfreq="$(cat /sys/bus/cpu/devices/cpu${cpu_array[0]}/cpufreq/cpuinfo_max_freq)"
minfreq="$(cat /sys/bus/cpu/devices/cpu${cpu_array[0]}/cpufreq/cpuinfo_min_freq)"
echo "restore CPU frequency to default"
cpupower frequency-set -u $maxfreq >/dev/null 2>&1
cpupower frequency-set -d $minfreq > /dev/null 2>&1


## C6=Enable to achieve maximum frequency
cpupower idle-set -E

NIDLE=`cpupower idle-info | awk -F ':' '/Number of idle states/ {print $2}'`
echo "Number of idle states is $NIDLE"

# - c-state control doesn't be effective on PCT!
cpupower -c 0,1 idle-set -D 50

# many GNR platform as a default enable of Hardware C1E
# set MSR1FC[bit1] = 0
if [[ $CPUVENDER == "GenuineIntel" ]]; then
    MSR1FC=$(rdmsr -p ${cpu_array[0]} 0x1fc)
    echo "CPU${cpu_array[0]} POWER_CTL(MSR 1FC) = $MSR1FC"
    newMSR1FC=$(echo "obase=16; $((0x$MSR1FC & ~(1 << 1)))" | bc)
    echo "Set CPU${cpu_array[0]} POWER_CTL(MSR 1FC) to 0x$newMSR1FC"
    wrmsr -p ${cpu_array[0]} 0x1fc 0x$newMSR1FC
    echo "CPU${cpu_array[0]} POWER_CTL(MSR 1FC) = $(rdmsr -p ${cpu_array[0]} 0x1fc)"

    MSR1FC=$(rdmsr -p ${cpu_array[1]} 0x1fc)
    newMSR1FC=$(echo "obase=16; $((0x$MSR1FC & ~(1 << 1)))" | bc)
    echo "Set CPU${cpu_array[1]} POWER_CTL(MSR 1FC) to 0x$newMSR1FC"
    wrmsr -p ${cpu_array[1]} 0x1fc 0x$newMSR1FC
    echo "CPU${cpu_array[1]} POWER_CTL(MSR 1FC) = $(rdmsr -p ${cpu_array[1]} 0x1fc)"
fi

WLDIR="/home/xtang/nccl-tests"
cp *.sh $WLDIR/
cd $WLDIR
NCCL_TEST="AllReduce"
GPU_NUM="1"

## CPU governor == powersave
cpupower frequency-set -g powersave > /dev/null 2>&1
echo "set CPU governor to powersave"

echo "Run kernel_launch on CPU ${cpu_array[0]}"
turbostat -c ${cpu_array[0]},${cpu_array[1]} -i2 -n 4 -q &
cmd1="taskset -c ${cpu_array[0]} ./nccl_test.sh ${GPU_NUM}  ${NCCL_TEST}"
echo $cmd1
eval $cmd1

echo "Run kernel_launch on CPU ${cpu_array[1]}"
turbostat -c ${cpu_array[0]},${cpu_array[1]} -i2 -n 4 -q &
cmd2="taskset -c ${cpu_array[1]} ./nccl_test.sh ${GPU_NUM} ${NCCL_TEST}"
echo $cmd2
eval $cmd2

## CPU governor == performance
cpupower frequency-set -g performance > /dev/null 2>&1
echo "Set CPU governor to performance"

echo "Run kernel_launch on CPU ${cpu_array[0]}"
turbostat -c ${cpu_array[0]},${cpu_array[1]} -i2 -n 4 -q &
cmd3="taskset -c ${cpu_array[0]} ./nccl_test.sh ${GPU_NUM} ${NCCL_TEST}"
echo $cmd3
eval $cmd3

echo "Run kernel_launch on CPU ${cpu_array[1]}"
turbostat -c ${cpu_array[0]},${cpu_array[1]} -i2 -n 4 -q &
cmd4="taskset -c ${cpu_array[1]} ./nccl_test.sh ${GPU_NUM} ${NCCL_TEST}"
echo $cmd4
eval $cmd4

echo "Run kernel_launch freq scaling on CPU ${cpu_array[0]}"
for freq in `seq 2200000 200000 $((maxfreq+100000))`
do
    sleep 10
    if [[ $freq > $maxfreq ]]; then
        freq=$maxfreq
    fi
    echo "Set ${cpu_array[0]} freq to $freq"
    # cpufreq doesn't work well in change CPU frequency.
    # Should I set both vCPU belonging to the same core?
    # user cpupowe frequency-set
    #echo $freq > /sys/bus/cpu/devices/cpu${cpu_array[0]}/cpufreq/scaling_min_freq
    #echo $freq > /sys/bus/cpu/devices/cpu${cpu_array[0]}/cpufreq/scaling_max_freq
    cpupower frequency-set -d $freq > /dev/null 2>&1
    cpupower frequency-set -u $freq > /dev/null 2>&1

    turbostat -c ${cpu_array[0]},${cpu_array[1]} -i2 -n5 -q &
    cmd5="taskset -c ${cpu_array[0]} ./nccl_test.sh ${GPU_NUM} ${NCCL_TEST}"
    mv nccl_results_results_rank1.csv nccl_results_rank1_${freq}.csv
    echo $cmd5
    eval $cmd5
done

cpupower frequency-set -d $minfreq > /dev/null 2>&1
cpupower frequency-set -u $maxfreq > /dev/null 2>&1

if [[ $CPUVENDER == "GenuineIntel" ]]; then
    ## Collect emon data
    cmd6="taskset -c ${cpu_array[0]} ./nccl_test.sh ${GPU_NUM} ${NCCL_TEST}"
    echo $cmd6
    #eval $cmd
    #pid=$!
    #sleep 2
    . /opt/intel/sep/sep_vars.sh
    sleep 2
    emon -collect-edp -f "emon-nccl-${NCCL_TEST}-$freq.dat" &
    eval $cmd6
    #sleep 60
    emon -stop
    #kill -9 $pid
fi

# collect nsys data
echo ""
echo "-----------------------------------------"
echo "set CPU${cpu_array[0]} frequency to $minfreq, and collect nsys data"
cpupower frequency-set -d $minfreq > /dev/null 2>&1
cpupower frequency-set -u $minfreq > /dev/null 2>&1

/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite true -o "nsys-nccl-test-${NCCL_TEST}-freq-$minfreq" \
    --stats=true taskset -c ${cpu_array[0]} ./nccl_test.sh ${GPU_NUM} ${NCCL_TEST}
echo ""
echo "-----------------------------------------"
echo "Set CPU${cpu_array[0]} frequency to $maxfreq, and collect nsys data"
cpupower frequency-set -d $maxfreq > /dev/null 2>&1
cpupower frequency-set -u $maxfreq > /dev/null 2>&1

/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite true -o "nsys-nccl-test-${NCCL_TEST}-freq-$maxfreq" \
    --stats=true taskset -c ${cpu_array[0]} ./nccl_test.sh ${GPU_NUM} ${NCCL_TEST}

cpupower frequency-set -d $minfreq > /dev/null 2>&1
cpupower frequency-set -u $maxfreq > /dev/null 2>&1

cpupower idle-set -E
echo ""
echo "Done"
echo ""
#OUTPUTFILE=kernel-launch-`hostname`-`date +%Y%m%d_%H%M`.zip
#zip -r dot_product-$OUTPUTFILE.zip *
