#!/bin/bash

set -e
if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export OPEN_BLAS_NUM_THREADS=1


# Install bipp in case of modifications
# -------------------------------------
sh install_bipp_on_izar.sh orliac
#sh install_pypeline_on_izar.sh

SPACK_SKA_ENV=bipp-izar-gcc
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh

VENV=VENV_IZARGCC
source $VENV/bin/activate

SOFT_DIR=$(pwd)

IN_DIR=/work/ska/longobs_eor_skalow/
MS_BASENAME=EOS_21cm-gf_202MHz_4h1d_1000.MS
MS_FILE=${IN_DIR}/${MS_BASENAME}
[ -d $MS_FILE ] || (echo "-E- MS dataset $MS_FILE not found" && exit 1)


# EOS_data.txt (/project/c31/SKA_low_images/eos_fits/EOS_data.txt)
# # z             nu [MHz]        dthet [deg]     FoV [deg]       mean_dT [mK]    std_dT [mK^2]
# 5.99969         202.92415       1.02011e-02     1.02011e+01     2.87925         4.38268
# 11.00101        118.35714       8.73413e-03     8.73413e+00     -1.56664        4.14754
# 14.99418        88.80768        8.22254e-03     8.22254e+00     -67.84012       16.31348

# FoV 10.2011 [deg] = 36723.96 [asec]
# With WSC_SIZE of 3072, scale is 12
# With WSC_SIZE of 4096, scale is 9

WSC_SIZE=3072; WSC_SCALE=5
WSC_SIZE=4096; WSC_SCALE=4
#WSC_SIZE=6144; WSC_SCALE=3
WSC_SIZE=512; WSC_SCALE=10
#WSC_SIZE=256; WSC_SCALE=10
#WSC_SIZE=1024; WSC_SCALE=5
TIME_SLICE_PE=10000
TIME_SLICE_IM=10000
SIGMA=0.9999

OUT_DIR=/work/ska/orliac/bipp-bench/michele_skalow_21cm/${WSC_SIZE}/${WSC_SCALE}
[ ! -d $OUT_DIR ] && mkdir -pv $OUT_DIR


RUN_WSCLEAN=1

WSCLEAN_OUT=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean
WSCLEAN_LOG=${OUT_DIR}/${MS_BASENAME}-dirty_wsclean.log

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
        -interval 0 1 \
        -name ${WSCLEAN_OUT} \
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

combs=('pypeline_ss_none' 'bipp_ss_cpu' 'bipp_ss_gpu')
combs=('bipp_ss_cpu' 'bipp_ss_gpu')
combs=('bipp_ss_gpu')
combs=('bipp_nufft_cpu')
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
  
  time python $py_script \
      --ms_file ${MS_FILE} \
      --telescope 'SKALOW' \
      --output_directory ${OUT_DIR} \
      --cluster izar \
      --processing_unit $proc_unit --compiler gcc \
      --precision double \
      --package ${package} \
      --nlev 4 \
      --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
      --sigma ${SIGMA} \
      --time_slice_pe ${TIME_SLICE_PE} --time_slice_im ${TIME_SLICE_IM} \
      --nufft_eps 0.001 \
      --algo ${algo} \
      --filter_negative_eigenvalues 0 \
      --wsc_log ${WSCLEAN_LOG} \
      #--debug
  
  python plots.py \
      --bb_grid  ${OUT_DIR}/I_lsq_eq_grid.npy \
      --bb_data  ${OUT_DIR}/I_lsq_eq_data.npy \
      --bb_json  ${OUT_DIR}/stats.json \
      --wsc_log  ${WSCLEAN_LOG} \
      --wsc_fits ${WSCLEAN_OUT}-dirty.fits \
      --flip_ud \
      --flip_lr \
      --outdir   ${OUT_DIR} \
      --outname  "skalow_21cm_${algo}_${package}_${proc_unit}"   ###### adapt here

done

deactivate 

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
