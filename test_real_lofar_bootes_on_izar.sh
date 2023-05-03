#!/bin/bash

set -e
if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export OPEN_BLAS_NUM_THREADS=1


# Install bipp in case of modifications
# -------------------------------------
#sh install_bipp_on_izar.sh orliac


SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh

source $VENV/bin/activate

MS_BASENAME=gauss4_t201806301100_SBL180.MS


WSC_SIZE=1024
WSC_SCALE=10
TIME_SLICE_PE=100
TIME_SLICE_IM=1000
SIGMA=0.9999

WSCLEAN_OUT=${MS_BASENAME}
WSCLEAN_LOG=${MS_BASENAME}.log
time wsclean \
    -verbose \
    -log-time \
    -channel-range 0 1 \
    -size ${WSC_SIZE} ${WSC_SIZE} \
    -scale ${WSC_SCALE}asec \
    -pol I \
    -weight natural \
    -niter 0 \
    -name ${WSCLEAN_OUT} \
    ${MS_BASENAME} \
    | tee ${WSCLEAN_LOG}


for package in 'pypeline'; do          # 'bipp' 'pypeline'

    for algo in 'ss'; do           # 'ss' 'nufft'

        for proc_unit in 'none'; do # 'none' 'cpu' 'gpu'

            if [[ $package == 'bipp' ]] && [[ $proc_unit == 'none' ]]; then
                echo "bipp + none => not possible, skip"
                continue
            fi
            
            py_script=real_lofar_bootes_${algo}_${package}.py

            echo "=================================================================================="
            echo  python $py_script $proc_unit
            echo "=================================================================================="

            time python $py_script \
                --ms_file ./${MS_BASENAME} \
                --output_directory $(pwd) \
                --cluster izar \
                --processing_unit $proc_unit --compiler gcc \
                --precision double \
                --package ${package} --nlev 1 \
                --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
                --sigma=${SIGMA} --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
                --nufft_eps=0.001 \
                --wsc_log ${MS_BASENAME}.log

            python plots.py \
                --bb_grid   I_lsq_eq_grid.npy \
                --bb_data   I_lsq_eq_data.npy \
                --bb_json   stats.json \
                --wsc_log   ${WSCLEAN_LOG} \
                --wsc_fits  ${WSCLEAN_OUT}-dirty.fits

        done
    done
done
