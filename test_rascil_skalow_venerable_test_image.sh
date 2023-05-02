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
#sh install_pypeline_on_izar.sh

SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh

source $VENV/bin/activate

SOFT_DIR=$(pwd)
MS_BASENAME=rascil_skalow_venerable_test_image.ms
OUT_DIR=$(pwd)/bb_dirty_maps/rascil_skalow_venerable
[ ! -d $OUT_DIR ] && mkdir -pv $OUT_DIR
IN_DIR=/work/backup/ska/orliac/rascil/
MS_FILE=${IN_DIR}/${MS_BASENAME}
[ -d $MS_FILE ] || (echo "-E- MS dataset $MS_FILE not found" && exit 1)


WSC_SIZE=256
WSC_SCALE=196

TIME_SLICE_PE=100
TIME_SLICE_IM=100
SIGMA=0.9999

RUN_WSCLEAN=1

if [[ $RUN_WSCLEAN == 1 ]]; then

    # dirty
    WSCLEAN_OUT=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean
    WSCLEAN_LOG=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean.log
    time wsclean \
        -verbose \
        -log-time \
        -channel-range 0 1 \
        -size ${WSC_SIZE} ${WSC_SIZE} \
        -scale ${WSC_SCALE}asec \
        -pol I \
        -weight natural \
        -niter 0 \
        -make-psf \
        -interval 0 1 \
        -name ${WSCLEAN_OUT} \
        ${MS_FILE} \
        | tee ${WSCLEAN_LOG}
    
    echo
    python ${SOFT_DIR}/wsclean_log_to_json.py --wsc_log ${WSCLEAN_LOG}
    echo
fi


for package in 'pypeline'; do          # 'bipp' 'pypeline'

    for algo in 'ss'; do           # 'ss' 'nufft'

        for proc_unit in 'none'; do # 'none' 'cpu' 'gpu'

            if [[ $package == 'bipp' ]] && [[ $proc_unit == 'none' ]]; then
                echo "bipp + none => not possible, skip"
                continue
            fi
            
            py_script=skalow_ms_${algo}_${package}.py

            echo "=================================================================================="
            echo  python $py_script $proc_unit
            echo "=================================================================================="

            time python $py_script \
                --ms_file ${MS_FILE} \
                --output_directory ${OUT_DIR} \
                --cluster izar \
                --processing_unit $proc_unit --compiler gcc \
                --precision double \
                --package ${package} --nlev 1 \
                --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
                --sigma=${SIGMA} --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
                --nufft_eps=0.001 \
                --wsc_log ${WSCLEAN_LOG}

            #exit 0

            python plots.py \
                --bb_grid  ${OUT_DIR}/I_lsq_eq_grid.npy \
                --bb_data  ${OUT_DIR}/I_lsq_eq_data.npy \
                --bb_json  ${OUT_DIR}/stats.json \
                --wsc_log  ${WSCLEAN_LOG} \
                --wsc_fits ${WSCLEAN_OUT}-dirty.fits \
                --flip_ud \
                --outdir   ${OUT_DIR} \
                --outname  'rascil_skalow_venerable'   ###### adapt here

        done
    done
done
