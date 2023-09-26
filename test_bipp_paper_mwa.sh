#!/bin/bash

set -e

if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
#export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
#export OPEN_BLAS_NUM_THREADS=1
echo "-I- OMP_NUM_THREADS = ${OMP_NUM_THREADS}"

set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   ## set -u : exit the script if you try to use an uninitialised variable
set -o errexit   ## set -e : exit the script if any statement returns a non-true return value

RUN_CASA=1
RUN_WSCLEAN=1
RUN_BIPP=1
RUN_BIPP_PLOT=1

PLOT_ONLY=0
if [[ $PLOT_ONLY == 1 ]]; then
    RUN_CASA=0; RUN_WSCLEAN=0; RUN_BIPP=0;
fi

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

SPACK_SKA_ENV=bipp-izar-gcc-dev
VENV=VENV_IZARGCC

# Keep hard coded for Slurm
SOFT_DIR=/home/orliac/SKA/epfl-radio-astro/bipp-bench

#IN_DIR=/work/ska/MWA
IN_DIR=/work/ska/MWA_sandbox
MS_BASENAME_=1133149192-187-188_Sun_10s_cal
MS_BASENAME=${MS_BASENAME_}.ms
MS_FILE=${IN_DIR}/${MS_BASENAME}
[ -d $MS_FILE ] || (echo "-E- MS dataset $MS_FILE not found" && exit 1)

CASA=/work/ska/soft/casa-6.5.3-28-py3.8/bin/casa

if [[ 1 == 0 ]]; then
    [ -f $CASA ] || (echo "Fatal. Could not find $CASA" && exit 1)
    #$CASA --version

    $CASA \
        --nogui \
        --norc \
        --notelemetry \
        --logfile abc123 \
        -c $SOFT_DIR/casa_set_uniform_weight.py \
        --ms_file ${MS_FILE}

    exit 1
fi


TELESCOPE="MWA"

TIME_START_IDX=0
TIME_END_IDX=20
TIME_SLICE_PE=1
TIME_SLICE_IM=1
TIME_TAG=${TIME_START_IDX}-${TIME_END_IDX}-${TIME_SLICE_PE}-${TIME_SLICE_IM}

WSC_SIZE=2048; WSC_SCALE=20

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source $VENV/bin/activate

FOV_DEG=$(python get_fov_deg.py --size $WSC_SIZE --scale $WSC_SCALE)

OUT_DIR=/work/ska/orliac/mwa_sandbox/${WSC_SIZE}/${WSC_SCALE}/${TIME_TAG}
[ ! -d $OUT_DIR ] && mkdir -pv $OUT_DIR


SIGMA=1.0

CHANNEL_ID=4

BIPP_NLEV=1 # Bluebild number of (positive) energy levels
BIPP_FNE=0  # Bluebild swith to filter out (=1) or not (=0) negative eigenvalues


WSCLEAN_OUT=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean
WSCLEAN_LOG=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean.log

CASA_OUT=${OUT_DIR}/${MS_BASENAME}-dirty_casa
CASA_LOG=${OUT_DIR}/${MS_BASENAME}-dirty_casa.log


if [[ $RUN_CASA == 1 ]]; then

    TIME_FILE=${OUT_DIR}/time.file

    python get_ms_timerange.py \
        --ms ${MS_FILE} \
        --data 'DATA' \
        --channel_id $CHANNEL_ID \
        --time_start_idx ${TIME_START_IDX} \
        --time_end_idx ${TIME_END_IDX} \
        --time_file ${TIME_FILE}
    #cat ${TIME_FILE}
    
    casa_start=$(sed -n '1p' ${TIME_FILE})
    casa_end=$(sed -n '2p' ${TIME_FILE})
    CASA_TIMERANGE="${casa_start}~${casa_end}"
    echo "CASA_TIMERANGE = $CASA_TIMERANGE"

    deactivate
    source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh

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
        --spw '*:'$CHANNEL_ID
    echo; echo

    source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
    source $VENV/bin/activate
fi

#        -make-psf \
#        -maxuvw-m ${MAXUVW_M} \
#        -gridder idg -idg-mode hybrid \
gridder_opt=''
#gridder_opt="-gridder idg -idg-mode cpu"
if [[ $RUN_WSCLEAN == 1 ]]; then
    time wsclean \
        -verbose \
        -log-time \
        -channel-range $CHANNEL_ID $(expr $CHANNEL_ID + 1) \
        -size ${WSC_SIZE} ${WSC_SIZE} \
        -scale ${WSC_SCALE}asec \
        -pol I \
        -weight natural \
        -name ${WSCLEAN_OUT} \
        -niter 0 \
        -interval ${TIME_START_IDX} ${TIME_END_IDX} \
        $gridder_opt \
        ${MS_FILE} \
        | tee ${WSCLEAN_LOG}
    
    echo
    python ${SOFT_DIR}/wsclean_log_to_json.py --wsc_log ${WSCLEAN_LOG}
    echo
fi


# Build list of combinations to run
# package:   'bipp', 'pypeline'
# algo:      'ss', 'nufft'
# proc_unit: 'cpu', 'gpu', 'none'
# ------------------------------------------------------------------------------
combs=('pypeline_ss_none' 'bipp_ss_cpu' 'bipp_ss_gpu' 'bipp_nufft_cpu' 'bipp_nufft_gpu')
combs=('bipp_ss_cpu' 'bipp_ss_gpu' 'bipp_nufft_cpu' 'bipp_nufft_gpu')
combs=('bipp_nufft_gpu' 'bipp_nufft_cpu' 'bipp_ss_gpu' 'bipp_ss_cpu')
combs=('bipp_nufft_gpu')

for comb in ${combs[@]}; do

  IFS=_ read -r package algo proc_unit <<< $comb
  if [[ $package == 'bipp' ]] && [[ $proc_unit == 'none' ]]; then
      echo "bipp + none => not possible, skip"
      continue
  fi

  BIPP_LOG=${OUT_DIR}/${comb}.log

  OUTNAME=${comb}_mwa

  py_script=ms_${algo}_${package}.py

  if [[ $RUN_BIPP == 1 ]]; then

      echo "=================================================================================="
      echo  python $py_script $proc_unit
      echo "=================================================================================="
        
     #export BIPP_LOG_LEVEL=DEBUG
      #--debug \
#          --maxuvw_m ${MAXUVW_M} \

      time python $py_script \
          --ms_file ${MS_FILE} \
          --telescope ${TELESCOPE} \
          --output_directory ${OUT_DIR} \
          --cluster izar \
          --processing_unit $proc_unit --compiler gcc \
          --precision single \
          --package ${package} \
          --nlev ${BIPP_NLEV} \
          --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
          --sigma ${SIGMA} \
          --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
          --time_start_idx ${TIME_START_IDX} --time_end_idx ${TIME_END_IDX} \
          --nufft_eps 0.00001 \
          --algo ${algo} \
          --filter_negative_eigenvalues ${BIPP_FNE} \
          --wsc_log ${WSCLEAN_LOG} \
          --channel_id ${CHANNEL_ID} \
          --outname ${OUTNAME} \
          |& tee ${BIPP_LOG}
  fi

  if [[ $RUN_BIPP_PLOT == 1 ]]; then

      sky_file=${IN_DIR}/${MS_BASENAME_}.sky
      sky_opt=''
      [ -f $sky_file ] && sky_opt="--sky_file $sky_file"
      echo sky_opt = $sky_opt
      
      outname="${TELESCOPE}_${algo}_${package}_${proc_unit}_${BIPP_NLEV}_${BIPP_FNE}_wsc_casa_bb"

      if [[ 1 == 1 ]]; then
          python plots.py \
                 --bb_grid  ${OUT_DIR}/${OUTNAME}_I_lsq_eq_grid.npy \
                 --bb_data  ${OUT_DIR}/${OUTNAME}_I_lsq_eq_data.npy \
                 --bb_json  ${OUT_DIR}/${OUTNAME}_stats.json \
                 --wsc_log  ${WSCLEAN_LOG} \
                 --wsc_fits ${WSCLEAN_OUT}-dirty.fits \
                 --casa_log  ${CASA_LOG} \
                 --casa_fits ${CASA_OUT}.image.fits \
                 --outdir   ${OUT_DIR} \
                 --wsc_size ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
                 $sky_opt \
                 --flip_ud \
                 --outname $outname
      fi
      
      if [ -f $sky_file ]; then
          python ./analysis/analyze_sky.py \
                 --sky_file ${OUT_DIR}/${outname}.sky \
                 --wsc_size ${WSC_SIZE} \
                 --wsc_scale ${WSC_SCALE} \
                 --outdir   ${OUT_DIR}
      fi

      echo "-I- plots to be found under ${OUT_DIR}"
  fi

done

deactivate 

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
