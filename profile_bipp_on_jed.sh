#!/bin/bash

set -e

if [[ -z "${OMP_NUM_THREADS}" ]]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export OPEN_BLAS_NUM_THREADS=1

SPACK_SKA_ENV=bipp-jed-gcc
VENV=VENV_JEDGCC
PACKAGE_PATH=/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed

source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/activate.sh
source $VENV/bin/activate

PYTHON=$(which python)
echo PYTHON = $PYTHON
$PYTHON -V

#module load intel-oneapi-vtune/2022.3.0

# Special from Pablo -- check permissions on system-wise profiling

VTUNE=0
ADVISOR=1

if [ $VTUNE == 1 ]; then
    /usr/lib64/ld-linux-x86-64.so.2 \
        /ssoft/spack/syrah/v1/opt/spack/linux-rhel8-x86_64_v2/gcc-8.5.0/intel-oneapi-vtune-2022.3.0-le7wqhah6taofs6p4rxo35ughsau6nhl/vtune/2022.3.0/bin64/vtune2 \
        -collect hotspots -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace \
        -source-search-dir=${PACKAGE_PATH}/src -search-dir=${PACKAGE_PATH}/src \
        -- $PYTHON lofar_bootes_ss_bipp.py --outdir . --cluster jed --processing_unit cpu --compiler gcc --precision double --package bipp --nsta 32 --nlev 8 --pixw 128
fi

PABLO=0

if [ $ADVISOR == 1 ]; then

    if [ $PABLO == 1 ]; then
        ADVISOR_CL="/usr/lib64/ld-linux-x86-64.so.2  /ssoft/spack/syrah/v1/opt/spack/linux-rhel8-x86_64_v2/gcc-8.5.0/intel-oneapi-advisor-2022.1.0-lyrpm6cewntekc2bblzvzoe6jmce2f56/advisor/2022.1.0/bin64/advisor2"
    else     
        module load intel-oneapi-advisor/2022.1.0
        module list
        ADVISOR_CL=advixe-cl
    fi

    #set +e
    #rm -r /tmp/abc123/*
    #set -e
    #LD_LIBRARY_PATH=/ssoft/spack/syrah/v1/opt/spack/linux-rhel8-x86_64_v2/gcc-8.5.0/intel-oneapi-advisor-2022.1.0-lyrpm6cewntekc2bblzvzoe6jmce2f56/advisor/2022.1.0/lib64:$LD_LIBRARY_PATH
    #python adv_self_check.py --log-dir=/tmp/abc123
    #exit 0
    
    $ADVISOR_CL --collect=roofline --enable-cache-simulation --no-profile-python \
                -project-dir /work/ska/orliac/profiling/bipp/jed/$$ --search-dir src:=${PACKAGE_PATH}/src \
                --strategy=ldconfig:notrace:notrace \
                --interval=5 \
                -- $PYTHON lofar_bootes_ss_bipp.py --outdir . --cluster jed --processing_unit cpu --compiler gcc --precision double --package bipp --nsta 60 --nlev 8 --pixw 256
    
    #module purge
fi

deactivate
source ~/SKA/ska-spack-env/${SPACK_SKA_ENV}/deactivate.sh
