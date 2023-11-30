#!/bin/bash

set -e

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source ../$VENV/bin/activate

export PYTHONPATH="..:$PYTHONPATH"

# Benchmark root folder
root5="/work/ska/papers/bipp/sim_skalow/benchmarks-5/oskar_9_point_sources_0-50-1-1_single_0"

# Select benchmark to analyse here
bench_root=${root5}

[ -d ${bench_root} ] || (echo "-E- bench_root ${bench_root} not found." && exit 1)

out_dir="/home/orliac/SKA/epfl-radio-astro/bipp-bench/analysis/plots_bench_v5/bb_sol_differential"
[ -d ${out_dir} ] || mkdir -pv ${out_dir}

python plot_bluebild_solutions_differential_plots.py \
       --bench_root ${bench_root} \
       --out_dir ${out_dir} \
       --sol1 'bipp/izar/gpu/cuda/single/nufft/0/1/' \
       --sol2 'bipp/izar/cpu/gcc/single/ss/0/1/'

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
