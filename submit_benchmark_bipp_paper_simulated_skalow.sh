#!/bin/bash

set -e

# Choose which repo to use
BIPP_REPO=orliac # upstream

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

if [[ $INSTALL_BIPP == 1 ]]; then
    sh install_bipp_on_izar.sh $BIPP_REPO
fi    
if [[ $INSTALL_PYPELINE == 1 ]]; then
    sh install_pypeline_on_izar.sh $BIPP_REPO
fi    


IN_DIR=/work/ska/papers/bipp/sim_skalow/data
MS_BASENAME=test64.MS
MS_FILE=${IN_DIR}/${MS_BASENAME}
[ -d $MS_FILE ] || (echo "-E- MS dataset $MS_FILE not found" && exit 1)


# ! Use "|" as a separtor between options
# ! Use "=" to pass value of Slurm option: --opt-name=value and not --opt-name value
# TODO(eo) White spaces are making problems, check that
# Dev: 1 socket / Prod: 2 sockets
#SLURM_OPTS="--partition=gpu|--gres=gpu:1|--time=00-12:00:00|--cpus-per-task=20|--mem=380G|--qos=scitas"
#SLURM_OPTS="--partition=test|--gres=gpu:1|--time=00-00:30:00|--cpus-per-task=20|--mem=150G|--qos=scitas"
SLURM_OPTS="--partition=gpu|--gres=gpu:2|--time=00-01:00:00|--cpus-per-task=40|--mem=360G|--qos=scitas"
#SLURM_OPTS="--partition=build|--gres=gpu:1|--time=00-01:00:00|--cpus-per-task=20|--mem=80G|--qos=scitas"
#SLURM_OPTS="--partition=debug|--gres=gpu:1|--time=00-01:00:00|--cpus-per-task=20|--mem=80G|--qos=scitas"

CLUSTER=izar

PIPELINE=ms # to be expanded as: ms_{algo}_{package}

TIME_START_IDX=0
TIME_END_IDX=50
TIME_SLICE_PE=1
TIME_SLICE_IM=1
TIME_TAG=${TIME_START_IDX}-${TIME_END_IDX}-${TIME_SLICE_PE}-${TIME_SLICE_IM}
PRECISION='single'
FNE=0
BENCH_NAME=${MS_BASENAME}_${TIME_TAG}_${PRECISION}_${FNE}

COMMON="--bench-name $BENCH_NAME"
COMMON+=" --outdir /work/ska/papers/bipp/sim_skalow/benchmarks"
COMMON+=" --pipeline $PIPELINE"
COMMON+=" --ms-file $MS_FILE"
COMMON+=" --cluster $CLUSTER"
COMMON+=" --time-start-idx ${TIME_START_IDX} --time-end-idx ${TIME_END_IDX}"
COMMON+=" --time-slice-pe ${TIME_SLICE_PE} --time-slice-im ${TIME_SLICE_IM}"
COMMON+=" --sigma 1.0"
#COMMON+=" --wsc_scale 4"
COMMON+=" --fov_deg 2"
COMMON+=" --precision ${PRECISION}"
COMMON+=" --telescope SKALOW"
COMMON+=" --filter_negative_eigenvalues ${FNE}"
COMMON+=" --channel_id 0"

echo $COMMON

# Reference (pypeline python cpu)
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package pypeline --proc-unit none --compiler gcc  --algo ss

# pypeline
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package pypeline --proc-unit gpu  --compiler cuda --algo ss
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package pypeline --proc-unit cpu  --compiler gcc  --algo ss 
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package pypeline --proc-unit gpu  --compiler cuda --algo nufft
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package pypeline --proc-unit cpu  --compiler gcc  --algo nufft

# bipp
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package bipp     --proc-unit gpu  --compiler cuda --algo ss
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package bipp     --proc-unit cpu  --compiler gcc  --algo ss
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package bipp     --proc-unit gpu  --compiler cuda --algo nufft
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" $COMMON --package bipp     --proc-unit cpu  --compiler gcc  --algo nufft

