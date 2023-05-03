#!/bin/bash

set -e
if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export OPEN_BLAS_NUM_THREADS=1

SPACK_SKA_ENV=bipp-jed-gcc
VENV=VENV_JEDGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source $VENV/bin/activate

python lofar_bootes_ss_bipp.py     --outdir . --cluster jed --processing_unit cpu --compiler gcc --precision double --package bipp --nsta 60 --nlev 60 --pixw 256

echo
echo "=================================================================================="
echo "=================================================================================="
echo

#python lofar_bootes_ss_pypeline.py --outdir . --cluster jed --processing_unit cpu --compiler gcc --precision double --package pypeline --nsta 60 --nlev 60 --pixw 256

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
