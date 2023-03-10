#!/bin/bash
#
# 1. Install using script: install_bipp_on_izar.sh
# 2. Run
#-------------------------------------------------------------------------------

set -e

MS_FILE=/home/orliac/SKA/epfl-radio-astro/bipp-bench/gauss4_t201806301100_SBL180.MS
[ -d $MS_FILE ] || (echo "-E- MS file $MS_FILE not found" && exit 1)

# ! Use "|" as a separtor between options
# ! Use "=" to pass value of Slurm option: --opt-name=value and not --opt-name value
# TODO(eo) White spaces are making problems, check that
# Dev: 1 socket / Prod: 2 sockets
SLURM_OPTS="--partition=test|--gres=gpu:1|--time=00-00:10:00|--cpus-per-task=20|--mem=80G"

CLUSTER=izar
BENCH_NAME=paper01
PIPELINE=real_lofar_bootes

COMMON="--bench-name $BENCH_NAME"
COMMON+=" --pipeline $PIPELINE"
COMMON+=" --ms-file $MS_FILE"
COMMON+=" --cluster $CLUSTER"
COMMON+=" --time-slice-pe 100 --time-slice-im 100"
COMMON+=" --sigma 1.0"
COMMON+=" --wsc_scale 10"
COMMON+=" --precision double"


# pypeline @ none, gpu, cpu
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit none --compiler gcc  --algo ss
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit gpu  --compiler cuda --algo ss
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit cpu  --compiler gcc  --algo ss 

# bipp | ss    | gpu
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON \
    --package bipp --proc-unit gpu --compiler cuda --algo ss
if [ 1 == 0 ]; then
    # bipp | ss    | cpu
    sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON \
        --package bipp --proc-unit cpu --compiler gcc --algo ss
    # bipp | nufft | gpu
    sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON \
        --package bipp --proc-unit gpu --compiler cuda --algo nufft
    # bipp | nufft | cpu
    sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON \
        --package bipp --proc-unit cpu --compiler gcc --algo nufft
fi
