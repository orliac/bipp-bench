#!/bin/bash

set -e

echo "-I- running $0"

if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
#export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
#export OPEN_BLAS_NUM_THREADS=1
echo "-I- OMP_NUM_THREADS   ${OMP_NUM_THREADS}"

set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   ## set -u : exit the script if you try to use an uninitialised variable
set -o errexit   ## set -e : exit the script if any statement returns a non-true return value

RUN_OSKAR=0
RUN_CASA=0
RUN_WSCLEAN=0
RUN_BIPP=1
RUN_BIPP_PLOT=1

WSC_SIZE=128;  WSC_SCALE=16

SIGMA=1.0
TIME_START_IDX=0
TIME_END_IDX=3
TIME_SLICE_PE=1
TIME_SLICE_IM=1
TIME_TAG=${TIME_START_IDX}-${TIME_END_IDX}-${TIME_SLICE_PE}-${TIME_SLICE_IM}
CHANNEL_ID_START=0
CHANNEL_ID_END=0
BIPP_NLEV=1 # Bluebild number of (positive) energy levels
BIPP_FNE=0  # Bluebild swith to filter out (=1) or not (=0) negative eigenvalues


INSTALL_BIPP=0
INSTALL_PYPELINE=0
precision=''
while getopts bpds flag
do
    case "${flag}" in
        b) INSTALL_BIPP=1;;
        p) INSTALL_PYPELINE=1;;
        d) precision='double';;
        s) precision='single';;
        ?) exit 1
    esac
done
echo "-I- INSTALL_BIPP?     $INSTALL_BIPP"
echo "-I- INSTALL_PYPELINE? $INSTALL_PYPELINE"
if [[ $precision != 'single' && $precision != 'double' ]]; then
   echo "-E- >$precision< not a valid precision; either single or double."
   exit 1
fi
echo "-I- precision?        $precision"

if [[ $RUN_BIPP == 1 && $INSTALL_BIPP == 1 ]]; then
    sh install_bipp_on_izar.sh --repo orliac #--dsk
fi    
if [[ $RUN_BIPP == 1 && $INSTALL_PYPELINE == 1 ]]; then
    sh install_pypeline_on_izar.sh orliac
fi

SPACK_SKA_ENV=bipp-izar-gcc-dev
VENV=VENV_IZARGCC

SOFT_DIR=/home/orliac/SKA/epfl-radio-astro/bipp-bench

# Gauss4
#IN_DIR=/home/orliac/SKA/epfl-radio-astro/bipp-bench
#MS_BASENAME_=gauss4_t201806301100_SBL180
#MS_BASENAME=${MS_BASENAME_}.MS
#TELESCOPE="LOFAR"

IN_DIR=/work/ska/papers/bipp/data/LEAP_paper
MS_BASENAME_=RX42_SB100-109.2ch10s
MS_BASENAME=${MS_BASENAME_}.ms
TELESCOPE="LOFAR"

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source $VENV/bin/activate


OUT_DIR=/work/ska/orliac/bipp_json_test_files/${WSC_SIZE}/${WSC_SCALE}/${TIME_TAG}-${precision}
[ ! -d $OUT_DIR ] && mkdir -pv $OUT_DIR


### Run OSKAR to produce a MS dataset
if [[ $RUN_OSKAR == 1 ]]; then

    MS_BASENAME_="oskar_9_sources"
    MS_BASENAME=${MS_BASENAME_}.ms
    TELESCOPE="SKALOW"
    IN_DIR=${OUT_DIR}/oskar

    ROOT_OSKAR=/home/orliac/SKA/oskar/OSKAR-2.8.3
    export OSKAR_INC_DIR=${ROOT_OSKAR}/inst/include
    export OSKAR_LIB_DIR=${ROOT_OSKAR}/inst/lib

    # Run if you need to install the Python interface of OSKAR
    #python -m pip install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'
    #python -m pip list

    [ ! -d $IN_DIR ] && mkdir -pv $IN_DIR

    INPUT_DIRECTORY="/work/ska/papers/bipp/oskar/tms"
    #TM="ska1low_new.tm"
    TM="bipp_tests_json.tm"
    #TM="lofar.tm"
    cp -r ${INPUT_DIRECTORY}/${TM} $IN_DIR

    read -r latlong < ${IN_DIR}/${TM}/position.txt
    IFS=' '
    read -a posarr <<< "$latlong"

    OSKAR_SIM=oskar_sim.py
    cp -v $OSKAR_SIM $IN_DIR
    
    echo "-I- IN_DIR = ${IN_DIR}"
    cd $IN_DIR
    #--single_source \
#            --phase_centre_ra_deg 260.0 \
#            --phase_centre_dec_deg 40.0 \

    python3 $IN_DIR/oskar_sim.py \
            --wsc_size $WSC_SIZE \
            --wsc_scale $WSC_SCALE \
            --num_time_steps $TIME_END_IDX \
            --input_directory $IN_DIR/${TM} \
            --telescope_lon ${posarr[0]} \
            --telescope_lat ${posarr[1]} \
            --single_source \
            --phase_centre_ra_deg 80.0 \
            --phase_centre_dec_deg -40.0 \
            --out_name $MS_BASENAME_
    cd -
fi


MS_FILE=${IN_DIR}/${MS_BASENAME}
[ -d $MS_FILE ] || (echo "-E- MS dataset $MS_FILE not found" && exit 1)


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
        --spw '*:'$CHANNEL_ID_START'~'$CHANNEL_ID_END
    echo; echo

    source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
    source $VENV/bin/activate
fi


if [[ $RUN_WSCLEAN == 1 ]]; then
    time wsclean \
        -verbose \
        -log-time \
        -channel-range $CHANNEL_ID_START  $(expr $CHANNEL_ID_END + 1) \
        -size ${WSC_SIZE} ${WSC_SIZE} \
        -scale ${WSC_SCALE}asec \
        -pol I \
        -weight natural \
        -name ${WSCLEAN_OUT} \
        -niter 0 \
        -make-psf \
        -interval ${TIME_START_IDX} ${TIME_END_IDX} \
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
combs=('pypeline_ss_none' 'bipp_ss_cpu' 'bipp_ss_gpu' 'bipp_nufft_cpu' 'bipp_nufft_gpu')


# List of filters to use separated with a "-" (underscore used in naming)
# STD SQRT INV INV_SQ SQ LSQ
INT_FILTERS="LSQ-STD-INV-SQRT"
SEN_FILTERS="INV_SQ"

for comb in ${combs[@]}; do

  IFS=_ read -r package algo proc_unit <<< $comb
  if [[ $package == 'bipp' ]] && [[ $proc_unit == 'none' ]]; then
      echo "bipp + none => not possible, skip"
      continue
  fi
  
  BIPP_LOG=${OUT_DIR}/${comb}.log
  [ -f ${BIPP_LOG} ] && rm -v ${BIPP_LOG}
  
  echo "-I- BIPP log ${BIPP_LOG}"
  OUTNAME=${comb}
  
  py_script=ms_${algo}_${package}.py
  
  if [[ $RUN_BIPP == 1 ]]; then
      #--debug
      #export BIPP_LOG_LEVEL=DEBUG
      echo "=================================================================================="
      echo  python $py_script $proc_unit $precision
      echo "=================================================================================="
      
      pwd
      
      time python $py_script \
           --ms_file ${MS_FILE} \
           --telescope ${TELESCOPE} \
           --output_directory ${OUT_DIR} \
           --cluster izar \
           --processing_unit $proc_unit --compiler gcc \
           --precision ${precision} \
           --package ${package} \
           --nlev ${BIPP_NLEV} \
           --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
           --sigma ${SIGMA} \
           --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
           --time_start_idx ${TIME_START_IDX} --time_end_idx ${TIME_END_IDX} \
           --nufft_eps 0.00001 \
           --algo ${algo} \
           --int_filters ${INT_FILTERS} \
           --sen_filters ${SEN_FILTERS} \
           --filter_negative_eigenvalues ${BIPP_FNE} \
           --wsc_log ${WSCLEAN_LOG} \
           --channel_id_start ${CHANNEL_ID_START} \
           --channel_id_end   $(expr $CHANNEL_ID_END + 1) \
           --outname ${OUTNAME} #\
          #2>&1 |& tee --append ${BIPP_LOG}
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
                 --flip_lr \
                 --outname $outname
      fi
      if [[ 1 == 0 ]]; then
          if [ -f $sky_file ]; then
              python ./analysis/analyze_sky.py \
                     --sky_file ${OUT_DIR}/${outname}.sky \
                     --wsc_size ${WSC_SIZE} \
                     --wsc_scale ${WSC_SCALE} \
                     --outdir   ${OUT_DIR}
          fi
      fi

      echo "-I- plots to be found under ${OUT_DIR}"
  fi

done

# To check the size of dumped json files
ls -rtl ${OUT_DIR}/*{input,output}*json || echo "-W- no input,output json files found"

deactivate 

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
