#!/bin/bash

set -e

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source ../$VENV/bin/activate

export PYTHONPATH="..:$PYTHONPATH"

#python analyze_benchmark.py \
    #--bench_root /work/ska/orliac/benchmarks/bipp-paper
    #--bench_root /work/ska/papers/bipp/sim_skalow/benchmarks/test64.MS \

# Benchmark root folder
#root1="/work/ska/papers/bipp/sim_skalow/benchmarks/test64.MS_0-50-1-1_single_0"
#root2="/work/ska/papers/bipp/sim_skalow/benchmarks-2/oskar_9_point_sources_0-50-1-1_single_0"
#root4="/work/ska/papers/bipp/sim_skalow/benchmarks-4/oskar_9_point_sources_0-50-1-1_single_0"
root5="/work/ska/papers/bipp/sim_skalow/benchmarks-5/oskar_9_point_sources_0-50-1-1_single_0"

# Select benchmark to analyse here
bench_root=${root5}

[ -d ${bench_root} ] || (echo "-E- bench_root ${bench_root} not found." && exit 1)

out_dir="/home/orliac/SKA/epfl-radio-astro/bipp-bench/analysis/plots_bench_v5"
[ -d ${out_dir} ] || mkdir -pv ${out_dir}


python analyze_benchmark.py \
    --bench_root ${bench_root} \
    --telescope 'SKALOW' \
    --out_dir ${out_dir}

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
