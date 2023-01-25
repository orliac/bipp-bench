#!/bin/bash

#SBATCH --array 0-0
#SBATCH --mem 80G
#SBATCH --cpus-per-task 36

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

input_file=jed_gcc.in
[[ -f $input_file ]] || (echo "-E- Input file >>$input_file<< not found." && exit 1)

source /home/orliac/SKA/ska-spack-env/bipp-jed-gcc/activate.sh
which python

#LURM_ARRAY_TASK_ID=2
SED_LINE_INDEX=$((${SLURM_ARRAY_TASK_ID}+1))

input_line=$(sed -n ${SED_LINE_INDEX}p $input_file | sed -e's/  */ /g')
echo "-I- Input line >>$input_line<<"

time python lofar_bootes_ss_cpp.py $input_line

#IFS=' ' read -ra  INPUT <<< "$input_line"
#for input in "${INPUT[@]}";
#do
#    echo "$input"
#done
#python lofar_bootes_ss_cpp.py $INPUT
source /home/orliac/SKA/ska-spack-env/bipp-jed-gcc/deactivate.sh
