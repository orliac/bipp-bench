#!/bin/bash

set -e

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source ../$VENV/bin/activate

export PYTHONPATH="..:$PYTHONPATH"

# Benchmark root folder
root5="/work/ska/papers/bipp/sim_skalow/benchmarks-5/oskar_9_point_sources_0-50-1-1_single_0"
root="/work/ska/orliac/debug/oskar_9s_SS_SP/256/4/0-1-1-1_single/"
root="/work/ska/orliac/debug/oskar_9s_SS_SP/256/4/0-50-1-1_single/"
root="/work/ska/orliac/debug/oskar_9s_SS_SP/256/4/0-25-1-1_single/"
root="/work/ska/orliac/debug/oskar_9s_SS_SP/256/4/0-100-1-1_single/"

root="/work/ska/orliac/debug/oskar_9s_SS_SP/256/4/0-100-1-1_single_gp2/"

# Select benchmark to analyse here
#bench_root=${root5}
bench_root=${root}

[ -d ${bench_root} ] || (echo "-E- bench_root ${bench_root} not found." && exit 1)

out_dir="/home/orliac/SKA/epfl-radio-astro/bipp-bench/analysis/plots_bench_v5/bb_sol_differential"
[ -d ${out_dir} ] || mkdir -pv ${out_dir}

root="/work/ska/orliac/debug/oskar_9s_SS_SP/256/4/"

#for gp in 'gp1' 'gp2' 'gp3' 'gp4'; do
for gp in 'gp3'; do
#    for nep in 1 25 50 100; do
    for nep in 400; do
        bench_root=$root/0-${nep}'-1-1_single'_${gp}
        python plot_bluebild_solutions_differential_plots.py \
               --bench_root $bench_root \
               --out_dir $root \
               --sol1 'SKALOW_nufft_bipp_gpu_1_0' \
               --sol2 'SKALOW_ss_bipp_cpu_1_0' \
               --ms_file 'oskar_9_sources.ms' \
               --outdir $root/plots \
               --outname "nufft_gpu_minus_ss_cpu_${gp}_${nep}"
        #       --path_sol1 'bipp/izar/gpu/cuda/single/nufft/0/1/' \
            #       --path_sol2 'bipp/izar/cpu/gcc/single/ss/0/1/'
    done
done
deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
