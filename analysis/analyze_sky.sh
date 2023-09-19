#!/bin/bash

set -e

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source ../$VENV/bin/activate

#python -m pip install seaborn

export PYTHONPATH="..:$PYTHONPATH"

#/work/ska/orliac/debug/oskar_bipp_paper/2048/4/0-600-1-1/SKALOW_nufft_bipp_gpu_1_0_wsc_casa_bb.sky

python analyze_sky.py \
       --sky_file /work/ska/orliac/debug/oskar_bipp_paper/2048/4/0-100-1-1/SKALOW_nufft_bipp_gpu_1_0_wsc_casa_bb.sky

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
