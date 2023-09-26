#!/bin/bash

set -e

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source ../$VENV/bin/activate

#python -m pip install seaborn

export PYTHONPATH="..:$PYTHONPATH"

#/work/ska/orliac/debug/oskar_bipp_paper/2048/4/0-600-1-1/SKALOW_nufft_bipp_gpu_1_0_wsc_casa_bb.sky

#python analyze_sky.py \
#       --sky_file /work/ska/orliac/debug/oskar_bipp_paper/2048/4/0-100-1-1/SKALOW_nufft_bipp_gpu_1_0_wsc_casa_bb.sky

root=/work/ska/papers/bipp/sim_skalow/benchmarks-2/oskar_9_point_sources_0-50-1-1_single_0
[ ! -d $root ] && (echo "-E- $root not found." && exit 1)

for dir in $(find $root -maxdepth 9 -type d ! -path "*__pycache__*" | grep -E "/256$|/512$|/1024$|/2048$"); do
    wsc_scale=4
    wsc_size=$(basename $dir)
    echo $wsc_scale $wsc_size
    for sky_file in $(ls $dir/*.sky); do
        echo $sky_file
        python analyze_sky.py \
               --sky_file  $sky_file \
               --wsc_scale $wsc_scale \
               --wsc_size  $wsc_size \
               --outdir    $dir
    done
done


deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
