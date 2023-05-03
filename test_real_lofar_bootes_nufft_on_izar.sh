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

sh ./install_bipp_on_izar.sh orliac

MS_FILE=gauss4_t201806301100_SBL180.MS

WSC_SIZE=1000
WSC_SCALE=10


if [ 1 == 0 ]; then
    for proc_unit in 'none' 'cpu' 'gpu'; do
        echo "=================================================================================="
        echo  python real_lofar_bootes_nufft_pypeline.py $proc_unit
        echo "=================================================================================="
        time python real_lofar_bootes_nufft_pypeline.py \
            --ms_file ${MS_FILE} \
            --outdir . \
            --cluster izar \
            --processing_unit $proc_unit --compiler gcc \
            --precision double \
            --package pypeline --nlev 1 \
            --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
            --sigma=0.9999 --time_slice_pe 200 --time_slice_im 2000
    done
fi

if [ 1 == 1 ]; then

    for iter in {1..200}; do
        
        echo; echo
        echo ">>>>>>>>>>>>>>>>>> iter $iter"
        echo
        #for proc_unit in 'cpu' 'gpu'; do
        for proc_unit in 'gpu'; do
            echo "=================================================================================="
            echo  python real_lofar_bootes_nufft_bipp.py $proc_unit
            echo "=================================================================================="
            time \
                #cuda-memcheck --leak-check full --log-file leak.log \
            #cuda-gdb --args \
            python real_lofar_bootes_nufft_bipp.py \
                --ms_file ${MS_FILE} \
                --output_directory /scratch/izar/orliac/test_nufft \
                --cluster izar \
                --processing_unit $proc_unit --compiler gcc \
                --precision double \
                --package bipp --nlev 4 \
                --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
                --sigma 0.9999 --time_slice_pe 100 --time_slice_im 1 \
                --wsc_log ${MS_FILE}_wsclean.log
        done
    done
fi

#python lofar_bootes_ss_pypeline.py --outdir . --cluster jed --processing_unit cpu --compiler gcc --precision double --package pypeline --nsta 60 --nlev 60 --pixw 256

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
