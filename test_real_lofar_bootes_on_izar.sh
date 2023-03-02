#!/bin/bash

set -e
if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export OPEN_BLAS_NUM_THREADS=1


# Install bipp in case of modifications
# -------------------------------------
sh install_bipp_on_izar.sh


SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh

source $VENV/bin/activate

MS_FILE=gauss4_t201806301100_SBL180.MS

WSC_SIZE=1000
WSC_SCALE=10
TIME_SLICE_PE=10
TIME_SLICE_IM=100
SIGMA=0.9999

for package in 'bipp'; do          # 'bipp' 'pypeline'

    for algo in 'nufft' 'ss'; do           # 'ss' 'nufft'

        for proc_unit in 'gpu'; do # 'none' 'cpu' 'gpu'

            if [[ $package == 'bipp' ]] && [[ $proc_unit == 'none' ]]; then
                echo "bipp + none => not possible, skip"
                continue
            fi
            
            py_script=real_lofar_bootes_${algo}_${package}.py

            echo "=================================================================================="
            echo  python $py_script $proc_unit
            echo "=================================================================================="

            time python $py_script \
                --ms_file ${MS_FILE} \
                --output_directory . \
                --cluster izar \
                --processing_unit $proc_unit --compiler gcc \
                --precision double \
                --package ${package} --nlev 4 \
                --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
                --sigma=${SIGMA} --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
                --nufft_eps=0.001 \
                --wsc_log ${MS_FILE}_wsclean.log
        done
    done
done
