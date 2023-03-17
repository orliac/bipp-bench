#!/bin/bash

set -e
if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export OPEN_BLAS_NUM_THREADS=1

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source $VENV/bin/activate

#python -m pip install --no-deps pycsou pyFFS
#git clone https://github.com/flatironinstitute/finufft.git
#cd finufft
#echo "CXXFLAGS += -DFFTW_PLAN_SAFE" > make.inc
#CXXFLAGS=-I${CMAKE_PREFIX_PATH}/include LDFLAGS="-L${CMAKE_PREFIX_PATH}/lib" make test -j
#make python
#cd -

python -m pip list | grep pypeline


MS_FILE=gauss4_t201806301100_SBL180.MS

WSC_SIZE=256 #1000
WSC_SCALE=10
TIME_SLICE_PE=100
TIME_SLICE_IM=100
SIGMA=0.9999

if [ 1 == 1 ]; then
    #for proc_unit in 'none' 'cpu' 'gpu'; do
    for proc_unit in 'cpu' 'gpu'; do
        echo "=================================================================================="
        echo  python real_lofar_bootes_ss_pypeline.py $proc_unit
        echo "=================================================================================="
        time python real_lofar_bootes_ss_pypeline.py \
            --ms_file ${MS_FILE} \
            --outdir . \
            --cluster izar \
            --processing_unit $proc_unit --compiler gcc \
            --precision double \
            --package pypeline --nlev 1 \
            --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
            --sigma=${SIGMA} --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
            --wsc_log ${MS_FILE}_wsclean.log
    done
fi

if [ 1 == 0 ]; then
    for proc_unit in 'cpu' 'gpu'; do
    #for proc_unit in 'gpu'; do
        echo "=================================================================================="
        echo  python real_lofar_bootes_ss_bipp.py $proc_unit
        echo "=================================================================================="
        time python real_lofar_bootes_ss_bipp.py \
            --ms_file ${MS_FILE} \
            --outdir . \
            --cluster izar \
            --processing_unit $proc_unit --compiler gcc \
            --precision double \
            --package bipp --nlev 4 \
            --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
            --sigma=${SIGMA} --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
            --wsc_log ${MS_FILE}_wsclean.log
    done
fi

#python lofar_bootes_ss_pypeline.py --outdir . --cluster jed --processing_unit cpu --compiler gcc --precision double --package pypeline --nsta 60 --nlev 60 --pixw 256

deactivate

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
