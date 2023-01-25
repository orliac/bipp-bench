#!/bin/bash

#SBATCH --array 0-1
#SBATCH --mem 80G
#SBATCH --cpus-per-task 20
#SBATCH --partition build
#SBATCH --gres gpu:1

# /!\ File expected to be copied where benchmark.in to be run resides

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

input_file=benchmark.in
[[ -f $input_file ]] || (echo "-E- Input file >>$input_file<< not found." && exit 1)

source /home/orliac/SKA/ska-spack-env/env-bipp-izar/activate.sh
which python

SED_LINE_INDEX=$((${SLURM_ARRAY_TASK_ID}+1))

input_line=$(sed -n ${SED_LINE_INDEX}p $input_file | sed -e's/  */ /g')
echo "-I- Input line >>$input_line<<"

time python lofar_bootes_ss_cpp.py $input_line

source /home/orliac/SKA/ska-spack-env/env-bipp-izar/deactivate.sh
