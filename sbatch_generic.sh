#!/bin/bash

set -e

echo
env | grep SLURM
echo



get_option_value() {
    option=$2
    opt_array=($1)
    len=${#opt_array[@]}
    for (( i=0; i<${len}; i++ )); do
        if [ "${opt_array[$i]}" == $2 ]; then
            echo ${opt_array[$i+1]}
        fi
    done
}

if ! options=$(getopt -u -o "" -l pipeline:,algo:,package:,cluster: -- "$@"); then
    exit 1
fi

set -- $options

while [ $# -gt 0 ]
do
    case $1 in
        --package)    package="$2";    shift;;
        --cluster)    cluster="$2";    shift;;
        --pipeline)   pipeline="$2";   shift;;
        --algo)       algo="$2";       shift;;
        (--) shift; break;;
        (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
        (*) break;;
    esac
    shift
done

: ${package:?Missing mandatory --package option to specify which package to run [bipp, pypeline]}
: ${cluster:?Missing mandatory --cluster option to specify the compiler [izar, jed]}
: ${pipeline:?Missing mandatory --pipeline option to specify the pipeline prefix}
: ${algo:?Missing mandatory --algo option to specify the algorithm [ss, nufft]}


# Note: this file is expected to be copied where benchmark.in to be run resides

echo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "-I- OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo
[ -x "$(command -v nvidia-smi)" ] && nvidia-smi
echo

input_file=benchmark.in
[[ -f $input_file ]] || (echo "-E- Input file >>$input_file<< not found." \
    && exit 1)

if [ $cluster == 'izar' ]; then
    ENV_SPACK="/home/orliac/SKA/ska-spack-env/bipp-izar-gcc"
    VENV="/home/orliac/SKA/epfl-radio-astro/bipp-bench/VENV_IZARGCC"
elif [ $cluster == 'jed' ]; then
    ENV_SPACK="/home/orliac/SKA/ska-spack-env/bipp-jed-gcc"
    VENV="/home/orliac/SKA/epfl-radio-astro/bipp-bench/VENV_JEDGCC"
else
    echo "-E- Unknown cluster $cluster"
    exit 1
fi
 
SOL_DIR=$(pwd)
echo "-I SOL_DIR = $SOL_DIR"

# Get option for current job in job array
SED_LINE_INDEX=$((${SLURM_ARRAY_TASK_ID}+1))
input_line=$(sed -n ${SED_LINE_INDEX}p $input_file | sed -e's/  */ /g')
echo "-I- Input line >>$input_line<<";
TIME_START_IDX="$(get_option_value "$input_line" "--time_start_idx")"
TIME_END_IDX="$(get_option_value "$input_line" "--time_end_idx")"
WSC_SIZE="$(get_option_value "$input_line" "--pixw")"
WSC_SCALE="$(get_option_value "$input_line" "--wsc_scale")"
MS_FILE="$(get_option_value "$input_line" "--ms_file")"
CHANNEL_ID="$(get_option_value "$input_line" "--channel_id")"
OUTNAME="$(get_option_value "$input_line" "--outname")"
OUT_DIR="$(get_option_value "$input_line" "--output_directory")"
echo "-I- OUT_DIR = $OUT_DIR"
[ ! -d $OUT_DIR ] && mkdir -pv $OUT_DIR
cd $OUT_DIR


SLURMOUT_BASENAME=slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
ln -s $SOL_DIR/${SLURMOUT_BASENAME} $OUT_DIR/${SLURMOUT_BASENAME}

## WARNING: CASA must be run before activating the environments!!
#           (conflits otherwise)


set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   ## set -u : exit the script if you try to use an uninitialised variable
set -o errexit   ## set -e : exit the script if any statement returns a non-true return value

RUN_OSKAR=1
RUN_CASA=1
RUN_WSCLEAN=1

CASA_OUT=dirty_casa
CASA_LOG=dirty_casa.log
CASA_TIME_LOG=dirty_casa_time.log

IN_DIR=${OUT_DIR}/oskar
sky_opt=''

### Run OSKAR to produce a MS dataset
if [[ $RUN_OSKAR == 1 ]]; then
    
    source ${ENV_SPACK}/activate.sh
    source ${VENV}/bin/activate

    MS_BASENAME_="oskar_9_sources"

    ROOT_OSKAR=/home/orliac/SKA/oskar/OSKAR-2.8.3
    export OSKAR_INC_DIR=${ROOT_OSKAR}/inst/include
    export OSKAR_LIB_DIR=${ROOT_OSKAR}/inst/lib

    # Run if you need to install the Python interface of OSKAR
    #python -m pip install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'
    #python -m pip list

    [ ! -d $IN_DIR ] && mkdir -pv $IN_DIR

    TM="ska1low_new.tm"
    INPUT_DIRECTORY="/work/ska/papers/bipp/oskar/tms"

    cp -r ${INPUT_DIRECTORY}/${TM} $IN_DIR

    read -r latlong < ${IN_DIR}/${TM}/position.txt
    IFS=' '
    read -a posarr <<< "$latlong"

    cp -v $SOL_DIR/oskar_sim.py $IN_DIR
    
    echo "-I- IN_DIR = ${IN_DIR}"
    cd $IN_DIR

    python3 $IN_DIR/oskar_sim.py \
            --wsc_size $WSC_SIZE \
            --wsc_scale $WSC_SCALE \
            --num_time_steps $TIME_END_IDX \
            --input_directory $IN_DIR/${TM} \
            --telescope_lon ${posarr[0]} \
            --telescope_lat ${posarr[1]} \
            --phase_centre_ra_deg 80.0 \
            --phase_centre_dec_deg -40.0 \
            --out_name $MS_BASENAME_ 
    cd -

    deactivate
    source ${ENV_SPACK}/deactivate.sh

    # Overwrite MS_FILE with OSKAR simulated data
    MS_BASENAME=${MS_BASENAME_}.ms
    ORIGINAL_MS_FILE=${IN_DIR}/${MS_BASENAME}
    MS_FILE=${OUT_DIR}/${MS_BASENAME}

    sky_file=${IN_DIR}/${MS_BASENAME_}.sky
    [ -f $sky_file ] && sky_opt="--sky_file $sky_file"
    echo "-I- sky_opt = $sky_opt"
    
else
    echo "-E- FIX ME"
    exit 1
    MS_BASENAME=$(basename -- "$MS_FILE")
    ORIGINAL_MS_FILE=$SOL_DIR/$MS_BASENAME
    MS_FILE=${OUT_DIR}/$MS_BASENAME
fi

[ -d $ORIGINAL_MS_FILE ] || (echo "-E- MS dataset $ORIGINAL_MS_FILE not found" && exit 1)
echo "-I- ORIGINAL_MS_FILE: $ORIGINAL_MS_FILE to be copied here . " `pwd` 


if [[ $RUN_CASA == 1 ]]; then

    echo =============================================================================
    echo CASA
    echo =============================================================================

    cp -rf  ${ORIGINAL_MS_FILE} ${MS_FILE}
    pwd
    ls -rtl
    
    echo ORIGINAL_MS_FILE = ${ORIGINAL_MS_FILE}
    echo MS_FILE = ${MS_FILE}
    echo MS_BASENAME = ${MS_BASENAME}

    source ${ENV_SPACK}/activate.sh
    source ${VENV}/bin/activate

    TIME_FILE=${OUT_DIR}/time.file
    python $SOL_DIR/get_ms_timerange.py \
        --ms ${MS_FILE} \
        --data 'DATA' \
        --channel_id ${CHANNEL_ID} \
        --time_start_idx ${TIME_START_IDX} \
        --time_end_idx ${TIME_END_IDX} \
        --time_file ${TIME_FILE}
    cat ${TIME_FILE}

    casa_start=$(sed -n '1p' ${TIME_FILE})
    casa_end=$(sed -n '2p' ${TIME_FILE})
    CASA_TIMERANGE="${casa_start}~${casa_end}"
    echo "CASA_TIMERANGE = $CASA_TIMERANGE"

    deactivate
    source ${ENV_SPACK}/deactivate.sh

    #CASA=~/SKA/casa-6.5.3-28-py3.8/bin/casa
    #[ -f $CASA ] || (echo "Fatal. Could not find $CASA" && exit 1)
    #which $CASA
    #$CASA --version

    CASA=/work/ska/soft/casa-6.5.3-28-py3.8/bin/casa
    [ -f $CASA ] || (echo "Fatal. Could not find $CASA" && exit 1)
    which $CASA
    $CASA --version
    
    casa_cmd="$CASA \
        --nogui \
        --norc \
        --notelemetry \
        --logfile ${CASA_LOG} \
        -c $SOL_DIR/casa_tclean.py \
        --ms_file ${MS_BASENAME} \
        --out_name ${CASA_OUT} \
        --imsize ${WSC_SIZE} \
        --cell ${WSC_SCALE} \
        --timerange ${CASA_TIMERANGE} \
        --spw *:${CHANNEL_ID}"
    echo
    echo "casa_cmd:"
    echo $casa_cmd
    echo

    /usr/bin/time \
        --output=$CASA_TIME_LOG \
        --portability \
        ${casa_cmd} | tee ${CASA_LOG}

    source ${ENV_SPACK}/activate.sh
    source ${VENV}/bin/activate

    echo
    python $SOL_DIR/casa_log_to_json.py --casa_log ${CASA_LOG}
    echo

    python $SOL_DIR/add_time_stats_to_bluebild_json.py \
        --bb_json ${CASA_OUT}.json \
        --bb_time_log ${CASA_TIME_LOG}

else
    source ${ENV_SPACK}/activate.sh
    source ${VENV}/bin/activate
fi


#-even-timesteps \
#-apply-primary-beam \

WSCLEAN_PSF="-make-psf"
WSCLEAN_OUT=dirty_wsclean
WSCLEAN_LOG=dirty_wsclean.log
WSCLEAN_TIME_LOG=dirty_wsclean_time.log
WSCLEAN_CLEAN_OUT=clean_wsclean
WSCLEAN_CLEAN_LOG=clean_wsclean.log

if [[ $RUN_WSCLEAN == 1 ]]; then

    echo =============================================================================
    echo WSClean
    echo =============================================================================

    # dirty

    cp -rf  ${ORIGINAL_MS_FILE} ${MS_FILE}

    wsclean_cmd="wsclean \
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
        ${MS_BASENAME}"

    /usr/bin/time \
        --output=$WSCLEAN_TIME_LOG \
        --portability \
        ${wsclean_cmd} | tee ${WSCLEAN_LOG}
    
    echo
    python $SOL_DIR/wsclean_log_to_json.py --wsc_log ${WSCLEAN_LOG}
    echo

    python $SOL_DIR/add_time_stats_to_bluebild_json.py \
          --bb_json ${WSCLEAN_OUT}.json \
          --bb_time_log ${WSCLEAN_TIME_LOG}

    # clean
    if [[ 1 == 0 ]]; then
        cp -r $SOL_DIR/$MS_BASENAME .
        time wsclean \
            -verbose \
            -log-time \
            -channel-range 0 1 \
            -size ${WSC_SIZE} ${WSC_SIZE} \
            -scale ${WSC_SCALE}asec \
            -pol I \
            -weight natural \
            -niter 0 \
            -name ${WSCLEAN_CLEAN_OUT} \
            $WSCLEAN_PSF \
            $WSCLEAN_INTERVAL \
            -niter 5000 -mgain 0.8 -threshold 0.1 \
            ${MS_BASENAME} \
            | tee ${WSCLEAN_CLEAN_LOG}
    fi
    #echo
    #python $SOL_DIR/wsclean_log_to_json.py --wsc_log ${WSCLEAN_CLEAN_LOG}
    #echo
fi

#cd -
cd ${SOL_DIR}

echo =============================================================================
echo ${pipeline}
echo =============================================================================
echo
BLUEBILD_LOG=$OUT_DIR/${OUTNAME}_bluebild.log
BLUEBILD_TIME_LOG=$OUT_DIR/${OUTNAME}_bluebild_time.log

export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1

echo "### input_line = " $input_line
input_line="${input_line} --ms_file ${MS_FILE}"
echo "### input_line = " $input_line

py_script=${pipeline}_${algo}_${package}.py

cp -rf  ${ORIGINAL_MS_FILE} ${MS_FILE}
    
/usr/bin/time \
    --output=$BLUEBILD_TIME_LOG \
    --portability python -u ${py_script} ${input_line} | tee ${BLUEBILD_LOG}

python $SOL_DIR/add_time_stats_to_bluebild_json.py \
    --bb_json $OUT_DIR/${OUTNAME}_stats.json \
    --bb_time_log ${BLUEBILD_TIME_LOG}

#time -p (cuda-gdb --args python -u ${pipeline}_${algo}_${package}.py $input_line | tee ${BLUEBILD_LOG}) 2> ${BLUEBILD_TIME_LOG}

echo
echo "cat ${BLUEBILD_TIME_LOG}"
echo "=========================================================================="
cat ${BLUEBILD_TIME_LOG}
echo "=========================================================================="


echo =============================================================================
echo Generate plots
echo =============================================================================
echo

cd $OUT_DIR

PLOTS_CMD=""
PLOTS_CMD+="python $SOL_DIR/plots.py"
PLOTS_CMD+=" --bb_grid   ${OUTNAME}_I_lsq_eq_grid.npy"
PLOTS_CMD+=" --bb_data   ${OUTNAME}_I_lsq_eq_data.npy"
PLOTS_CMD+=" --bb_json   ${OUTNAME}_stats.json"
PLOTS_CMD+=" --outdir ${OUT_DIR}"
PLOTS_CMD+=" --outname ${OUTNAME}"
PLOTS_CMD+=" --flip_lr"
PLOTS_CMD+=" --wsc_fits  ${WSCLEAN_OUT}-dirty.fits"
PLOTS_CMD+=" --wsc_log   ${WSCLEAN_LOG}"
PLOTS_CMD+=" --casa_fits ${CASA_OUT}.image.fits"
PLOTS_CMD+=" --casa_log  ${CASA_LOG}"
PLOTS_CMD+=" --wsc_size ${WSC_SIZE} --wsc_scale ${WSC_SCALE}"
PLOTS_CMD+=" ${sky_opt}"
$PLOTS_CMD

cd -

deactivate
source ${ENV_SPACK}/deactivate.sh
