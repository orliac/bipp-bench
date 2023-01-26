#!/bin/bash
#
# 1. Install
# 2. Run
#-------------------------------------------------------------------------------

set -e

# ! Use "|" as a separtor between options
# ! Use "=" to pass value of Slurm option: --opt-name=value and not --opt-name value
# White spaces are making problems, check that
SLURM_OPTS_IZAR="--partition=gpu|--gres=gpu:1|--time=00-00:15:00|--cpus-per-task=20|--array=0-1"

sh submit_benchmarks_generic.sh --slurm-opts "$SLURM_OPTS_IZAR" --bench-name abc --package pypeline --cluster izar --proc-unit none --compiler gcc
#sh submit_benchmarks_generic.sh --bench-name abc --package pypeline --cluster izar --proc-unit gpu  --compiler cuda 
#sh submit_benchmarks_generic.sh --bench-name abc --package pypeline --cluster izar --proc-unit cpu  --compiler gcc
