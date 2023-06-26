#!/bin/bash

set -e
if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export OPEN_BLAS_NUM_THREADS=1
echo "-I- OMP_NUM_THREADS = ${OMP_NUM_THREADS}"

RUN_CASA=0
RUN_WSCLEAN=0
RUN_BIPP=1

INSTALL_BIPP=0
INSTALL_PYPELINE=0
while getopts bp flag
do
    case "${flag}" in
        b) INSTALL_BIPP=1;;
        p) INSTALL_PYPELINE=1;;
    esac
done
echo "INSTALL_BIPP?     $INSTALL_BIPP"
echo "INSTALL_PYPELINE? $INSTALL_PYPELINE"

if [[ $RUN_BIPP == 1 && $INSTALL_BIPP == 1 ]]; then
    sh install_bipp_on_izar.sh orliac
fi    
if [[ $RUN_BIPP == 1 && $INSTALL_PYPELINE == 1 ]]; then
    sh install_pypeline_on_izar.sh orliac
fi    


SPACK_SKA_ENV=bipp-izar-gcc
VENV=VENV_IZARGCC

SOFT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
IN_DIR=${SOFT_DIR}

MS_BASENAME=gauss4_t201806301100_SBL180.MS
MS_FILE=${IN_DIR}/${MS_BASENAME}
[ -d $MS_FILE ] || (echo "-E- MS dataset $MS_FILE not found" && exit 1)

WSC_SIZE=512; WSC_SCALE=9

SIGMA=0.9999

TIME_START_IDX=0
TIME_END_IDX=1
TIME_SLICE_PE=1
TIME_SLICE_IM=1
TIME_TAG=${TIME_START_IDX}-${TIME_END_IDX}-${TIME_SLICE_PE}-${TIME_SLICE_IM}

OUT_DIR=/work/ska/orliac/bipp/debug/gauss4/${WSC_SIZE}/${WSC_SCALE}/${TIME_TAG}
[ ! -d $OUT_DIR ] && mkdir -pv $OUT_DIR

WSCLEAN_OUT=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean
WSCLEAN_LOG=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean.log

CASA_OUT=${OUT_DIR}/${MS_BASENAME}-dirty_casa
CASA_LOG=${OUT_DIR}/${MS_BASENAME}-dirty_casa.log

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source $VENV/bin/activate

if [[ $RUN_CASA == 1 ]]; then

    TIME_FILE=${OUT_DIR}/time.file

    python get_ms_timerange.py \
        --ms ${MS_FILE} \
        --data 'DATA' \
        --channel_id 0 \
        --time_start_idx ${TIME_START_IDX} \
        --time_end_idx ${TIME_END_IDX} \
        --time_file ${TIME_FILE}
    echo ${TIME_START_IDX} ${TIME_END_IDX}
    cat ${TIME_FILE}
    #exit 1

    casa_start=$(sed -n '1p' ${TIME_FILE})
    casa_end=$(sed -n '2p' ${TIME_FILE})
    CASA_TIMERANGE="${casa_start}~${casa_end}"
    echo "CASA_TIMERANGE = $CASA_TIMERANGE"

    deactivate
    source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh

    CASA=/work/ska/soft/casa-6.5.3-28-py3.8/bin/casa
    [ -f $CASA ] || (echo "Fatal. Could not find $CASA" && exit 1)
    $CASA --version

    $CASA \
        --nogui \
        --norc \
        --notelemetry \
        --logfile ${CASA_LOG} \
        -c $SOFT_DIR/casa_tclean.py \
        --ms_file ${MS_FILE} \
        --out_name ${CASA_OUT} \
        --imsize ${WSC_SIZE} \
        --cell ${WSC_SCALE} \
        --timerange  $CASA_TIMERANGE\
        --spw '*:0'
    echo; echo

    source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
    source $VENV/bin/activate
fi

#exit 1

if [[ $RUN_WSCLEAN == 1 ]]; then
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
        -interval ${TIME_START_IDX} ${TIME_END_IDX} \
        -name ${WSCLEAN_OUT} \
        ${MS_FILE} \
        | tee ${WSCLEAN_LOG}
    
    echo
    echo WSCLEAN_LOG: ${WSCLEAN_LOG}
    python ${SOFT_DIR}/wsclean_log_to_json.py --wsc_log ${WSCLEAN_LOG}
    echo
fi


# Build list of combinations to run
# package:   'bipp', 'pypeline'
# algo:      'ss', 'nufft'
# proc_unit: 'cpu', 'gpu', 'none'

combs=('pypeline_ss_none' 'bipp_ss_cpu' 'bipp_ss_gpu')
combs=('bipp_ss_cpu' 'bipp_ss_gpu')
combs=('bipp_ss_cpu' 'bipp_nufft_cpu')
combs=('bipp_nufft_gpu')
#combs=('bipp_nufft_gpu')
#combs=('bipp_nufft_cpu' 'bipp_nufft_gpu')
#combs=('pypeline_ss_none' 'bipp_ss_cpu' 'bipp_ss_gpu' 'bipp_nufft_cpu' 'bipp_nufft_gpu')
combs=('bipp_nufft_gpu')

for comb in ${combs[@]}; do

  IFS=_ read -r package algo proc_unit <<< $comb
  if [[ $package == 'bipp' ]] && [[ $proc_unit == 'none' ]]; then
      echo "bipp + none => not possible, skip"
      continue
  fi
  
  py_script=ms_${algo}_${package}.py

  echo "=================================================================================="
  echo  python $py_script $proc_unit
  echo "=================================================================================="
  
  if [[ $RUN_BIPP == 1 ]]; then
      time python $py_script \
          --ms_file ${MS_FILE} \
          --telescope 'LOFAR' \
          --output_directory ${OUT_DIR} \
          --cluster izar \
          --processing_unit $proc_unit --compiler gcc \
          --precision double \
          --package ${package} \
          --nlev 1 \
          --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
          --sigma ${SIGMA} \
          --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
          --time_start_idx ${TIME_START_IDX} --time_end_idx ${TIME_END_IDX} \
          --nufft_eps 0.001 \
          --algo ${algo} \
          --filter_negative_eigenvalues 0 \
          --wsc_log ${WSCLEAN_LOG} \
          #--debug
  fi

  if [[ 1 == 1 ]]; then
      python plots.py \
          --bb_grid  ${OUT_DIR}/I_lsq_eq_grid.npy \
          --bb_data  ${OUT_DIR}/I_lsq_eq_data.npy \
          --bb_json  ${OUT_DIR}/stats.json \
          --wsc_log  ${WSCLEAN_LOG} \
          --wsc_fits ${WSCLEAN_OUT}-dirty.fits \
          --casa_log  ${CASA_LOG} \
          --casa_fits ${CASA_OUT}.image.fits \
          --flip_ud \
          --flip_lr \
          --outdir   ${OUT_DIR} \
          --outname  "gauss4_${algo}_${package}_${proc_unit}"   ###### adapt here

      echo "-I- plots to be found under ${OUT_DIR}"
  fi

done

deactivate 

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
