#!/bin/bash

set -e

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source ../$VENV/bin/activate

export PYTHONPATH="..:$PYTHONPATH"

#python analyze_benchmark.py \
#    --bench_root /work/ska/orliac/benchmarks/bipp-paper

    #--bench_root /work/ska/papers/bipp/sim_skalow/benchmarks/test64.MS \

python analyze_benchmark.py \
    --bench_root /work/ska/papers/bipp/sim_skalow/benchmarks/test64.MS_0-50-1-1_single_0 \
    --telescope 'SKALOW'

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
