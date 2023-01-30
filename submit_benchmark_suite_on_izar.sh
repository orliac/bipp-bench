#!/bin/bash
#
# 1. Install
# 2. Run
#-------------------------------------------------------------------------------

set -e

# ! Use "|" as a separtor between options
# ! Use "=" to pass value of Slurm option: --opt-name=value and not --opt-name value
# White spaces are making problems, check that
SLURM_OPTS_IZAR="--partition=gpu|--gres=gpu:2|--time=03-00:00:00|--cpus-per-task=40|--mem=180G|--array=0-195"

BENCH_NAME=bench00

# pypeline @ none, gpu, cpu
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS_IZAR" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit none --compiler gcc  --algo ss
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS_IZAR" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit gpu  --compiler cuda --algo ss
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS_IZAR" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit cpu  --compiler gcc  --algo ss 
# bipp @ gpu, cpu
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS_IZAR" --bench-name $BENCH_NAME --package bipp     --cluster izar --proc-unit gpu  --compiler cuda --algo ss
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS_IZAR" --bench-name $BENCH_NAME --package bipp     --cluster izar --proc-unit cpu  --compiler gcc  --algo ss
