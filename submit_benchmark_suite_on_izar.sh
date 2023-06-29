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


#MS_FILE=/home/orliac/SKA/epfl-radio-astro/bipp-bench/gauss4_t201806301100_SBL180.MS
#MS_FILE=gauss4_t201806301100_SBL180.MS
#MS_FILE=/work/ska/orliac/rascil/rascil_skalow_venerable_test_image.ms
IN_DIR=/work/ska/orliac/RADIOBLOCKS
MS_BASENAME=EOS_21cm_202MHz_10min_1000.MS
MS_FILE=${IN_DIR}/${MS_BASENAME}
[ -d $MS_FILE ] || (echo "-E- MS dataset $MS_FILE not found" && exit 1)


# ! Use "|" as a separtor between options
# ! Use "=" to pass value of Slurm option: --opt-name=value and not --opt-name value
# TODO(eo) White spaces are making problems, check that
# Dev: 1 socket / Prod: 2 sockets
#SLURM_OPTS="--partition=gpu|--gres=gpu:1|--time=00-12:00:00|--cpus-per-task=20|--mem=400G"
#SLURM_OPTS="--partition=test|--gres=gpu:1|--time=00-00:30:00|--cpus-per-task=20|--mem=150G"
SLURM_OPTS="--partition=gpu|--gres=gpu:1|--time=00-01:00:00|--cpus-per-task=20|--mem=80G"

CLUSTER=izar
BENCH_NAME=bipp-paper
PIPELINE=ms # to be expanded as: ms_{algo}_{package}

COMMON="--bench-name $BENCH_NAME"
COMMON+=" --pipeline $PIPELINE"
COMMON+=" --ms-file $MS_FILE"
COMMON+=" --cluster $CLUSTER"
COMMON+=" --time-start-idx 0 --time-end-idx 10"
COMMON+=" --time-slice-pe 1 --time-slice-im 1"
COMMON+=" --sigma 1.0"
COMMON+=" --wsc_scale 4"
COMMON+=" --precision double"
COMMON+=" --telescope SKALOW"
COMMON+=" --filter_negative_eigenvalues 0"

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

