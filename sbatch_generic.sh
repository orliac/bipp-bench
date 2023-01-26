#!/bin/bash

set -e

if ! options=$(getopt -u -o "" -l package:,cluster: -- "$@"); then
    exit 1
fi

set -- $options

while [ $# -gt 0 ]
do
    case $1 in
        --package)    package="$2";    shift;;
        --cluster)    cluster="$2" ;   shift;;
        (--) shift; break;;
        (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
        (*) break;;
    esac
    shift
done

: ${package:?Missing mandatory --package option to specify which package to run [bipp, pypeline]}
: ${cluster:?Missing mandatory --cluster option to specify the compiler [izar, jed]}


# Note: this file is expected to be copied where benchmark.in to be run resides

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

input_file=benchmark.in
[[ -f $input_file ]] || (echo "-E- Input file >>$input_file<< not found." \
    && exit 1)

if [ $cluster == 'izar' ]; then
    ENV_SPACK="/home/orliac/SKA/ska-spack-env/env-bipp-izar"
elif [ $cluster == 'jed' ]; then
    ENV_SPACK="/home/orliac/SKA/ska-spack-env/bipp-jed-gcc"
else
    echo "-E- Unknown cluster $cluster"
    exit 1
fi
 
source ${ENV_SPACK}/activate.sh

SED_LINE_INDEX=$((${SLURM_ARRAY_TASK_ID}+1))

input_line=$(sed -n ${SED_LINE_INDEX}p $input_file | sed -e's/  */ /g')
echo "-I- Input line >>$input_line<<"

time python lofar_bootes_ss_${package}.py $input_line

source ${ENV_SPACK}/deactivate.sh
