#!/bin/bash
#
# 1. Install using script: install_bipp_on_izar.sh
# 2. Run
#-------------------------------------------------------------------------------

set -e

# ! Use "|" as a separtor between options
# ! Use "=" to pass value of Slurm option: --opt-name=value and not --opt-name value
# White spaces are making problems, check that
#SLURM_OPTS="--partition=gpu|--gres=gpu:1|--time=00-06:00:00|--cpus-per-task=20|--mem=80G|--array=0-8%4"
SLURM_OPTS="--partition=gpu|--gres=gpu:1|--time=00-00:15:00|--cpus-per-task=20|--mem=80G|--array=0-59%4"
#SLURM_OPTS="--partition=gpu|--gres=gpu:2|--time=00-00:15:00|--cpus-per-task=40|--mem=0|--array=0-59%4"
#SLURM_OPTS="--partition=build|--gres=gpu:1|--time=00-00:15:00|--cpus-per-task=20|--mem=80G|--array=0-59%1"

BENCH_NAME=bench04_72
SLURM_OPTS="--partition=gpu|--gres=gpu:2|--time=00-00:10:00|--cpus-per-task=40|--mem=0"

# pypeline @ none, gpu, cpu
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit none --compiler gcc  --algo ss
sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit gpu  --compiler cuda --algo ss
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package pypeline --cluster izar --proc-unit cpu  --compiler gcc  --algo ss 

# bipp @ gpu
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package bipp     --cluster izar --proc-unit gpu  --compiler cuda --algo ss
# bipp @ cpu
#sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS" --bench-name $BENCH_NAME --package bipp     --cluster izar --proc-unit cpu  --compiler gcc  --algo ss
