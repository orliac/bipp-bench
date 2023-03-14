#!/bin/bash

set -e

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
echo SOL_DIR = $SOL_DIR

# Get option for current job in job array
SED_LINE_INDEX=$((${SLURM_ARRAY_TASK_ID}+1))
input_line=$(sed -n ${SED_LINE_INDEX}p $input_file | sed -e's/  */ /g')
echo "-I- Input line >>$input_line<<"; echo
WSC_SIZE="$(get_option_value "$input_line" "--pixw")"
WSC_SCALE="$(get_option_value "$input_line" "--wsc_scale")"
MS_FILE="$(get_option_value "$input_line" "--ms_file")"
OUT_DIR="$(get_option_value "$input_line" "--output_directory")"
[ ! -d $OUT_DIR ] && mkdir -pv $OUT_DIR
cd $OUT_DIR


## WARNING: CASA must be run before activating the environments!!
#           (conflits otherwise)

echo =============================================================================
echo CASA
echo =============================================================================
CASA=~/SKA/casa-6.5.3-28-py3.8/bin/casa
[ -f $CASA ] || (echo "Fatal. Could not find $CASA" && exit 1)
which $CASA
#$CASA --help
CASA_OUT=dirty_casa
CASA_LOG=casa.log
$CASA \
    --nogui \
    --norc \
    --notelemetry \
    --logfile ${CASA_LOG} \
    -c $SOL_DIR/casa_tclean.py \
    --ms_file ${MS_FILE} \
    --out_name ${CASA_OUT} \
    --imsize ${WSC_SIZE} \
    --cell ${WSC_SCALE} \
    --spw '*:0'
echo; echo

source ${ENV_SPACK}/activate.sh
source ${VENV}/bin/activate

echo
python $SOL_DIR/casa_log_to_json.py --casa_log ${CASA_LOG}
echo

echo
echo =============================================================================
echo WSClean
echo =============================================================================
echo
WSCLEAN_OUT=dirty_wsclean
WSCLEAN_LOG=wsclean.log
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
    ${MS_FILE} \
    | tee ${WSCLEAN_LOG}

echo
python $SOL_DIR/wsclean_log_to_json.py --wsc_log ${WSCLEAN_LOG}
echo

cd -

echo =============================================================================
echo ${pipeline}
echo =============================================================================
echo
time python ${pipeline}_${algo}_${package}.py $input_line
echo

echo =============================================================================
echo Generate plots
echo =============================================================================
echo
cd $OUT_DIR
python $SOL_DIR/plots.py \
    --bb_grid   I_lsq_eq_grid.npy \
    --bb_data   I_lsq_eq_data.npy \
    --bb_json   stats.json \
    --wsc_fits  ${WSCLEAN_OUT}-dirty.fits \
    --wsc_log   ${WSCLEAN_LOG} \
    --casa_fits ${CASA_OUT}.image.fits \
    --casa_log  ${CASA_LOG}

cd -

deactivate
source ${ENV_SPACK}/deactivate.sh
