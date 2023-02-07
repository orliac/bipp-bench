#!/bin/bash

set -e

export OMP_NUM_THREADS=20
export OPEN_BLAS_NUM_THREADS=1

SPACK_SKA_ENV=env-bipp-izar
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source $VENV/bin/activate

#python lofar_bootes_ss_bipp.py --outdir . --cluster izar --processing_unit cpu --compiler gcc --precision double --package bipp --nsta 60 --nlev 60 --pixw 256
echo
echo "=================================================================================="
echo "=================================================================================="
echo
python lofar_bootes_ss_bipp.py --outdir . --cluster izar --processing_unit gpu --compiler cuda --precision double --package bipp --nsta 120 --nlev 60 --pixw 256
echo
echo "=================================================================================="
echo "=================================================================================="
echo
#python lofar_bootes_ss_pypeline.py --outdir . --cluster izar --processing_unit cpu --compiler gcc --precision double --package pypeline --nsta 60 --nlev 60 --pixw 256
echo
echo "=================================================================================="
echo "=================================================================================="
echo
sh install_pypeline_on_izar.sh
#cuda-gdb --args
#cuda-memcheck --leak-check full --log-file leak.log \
#python lofar_bootes_ss_pypeline.py --outdir . --cluster izar --processing_unit gpu --compiler cuda --precision double --package bipp --nsta 60 --nlev 60 --pixw 4096
#python lofar_bootes_ss_pypeline.py --outdir . --cluster izar --processing_unit gpu --compiler cuda --precision double --package pypeline --nsta 60 --nlev 60 --pixw 256
#python lofar_bootes_ss_pypeline.py --outdir . --cluster izar --processing_unit cpu --compiler gcc --precision double --package pypeline --nsta 60 --nlev 16 --pixw 256

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
