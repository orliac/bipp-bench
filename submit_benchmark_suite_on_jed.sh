#!/bin/bash
#
# 1. Install using script: install_bipp_on_izar.sh
# 2. Run
#-------------------------------------------------------------------------------

set -e

export NUMEXPR_MAX_THREADS=72  ### Number of physical core


# ! Use "|" as a separtor between options
# ! Use "=" to pass value of Slurm option: --opt-name=value and not --opt-name value
# White spaces are making problems, check that
SLURM_OPTS="--partition=standard|--time=00-01:00:00|--cpus-per-task=72|--mem=0"

BENCH_NAME=bench04_72
SLURM_OPTS="--partition=standard|--time=00-12:00:00|--cpus-per-task=72|--mem=0"

# pypeline @ none, cpu
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster jed --proc-unit none --compiler gcc  --algo ss
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster jed --proc-unit cpu  --compiler gcc  --algo ss 

# bipp @ cpu
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package bipp     --cluster jed --proc-unit cpu  --compiler gcc  --algo ss
