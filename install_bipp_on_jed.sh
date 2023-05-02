#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

CLUSTER=jed
PACKAGE=bipp

PACKAGE_NAME="${PACKAGE}-${CLUSTER}"
PACKAGE_ROOT=${SCRIPT_DIR}/${PACKAGE_NAME}
echo PACKAGE_ROOT = ${PACKAGE_ROOT}

# Delete if exists already + Git clone 
if [ 1 == 0 ]; then
    [ -d $PACKAGE_ROOT ] && rm -rf ${PACKAGE_ROOT}
    git clone https://github.com/epfl-radio-astro/${PACKAGE}.git ${PACKAGE_ROOT}
fi

[ -d $PACKAGE_ROOT ] || (echo "-E- ${PACKAGE_ROOT} directory does not exist" && exit 1)

# Activate Spack environment for izar
source ~/SKA/ska-spack-env/bipp-jed-gcc/activate.sh
python -V

# Just in case, remove bipp from outside any virtual env
python -m pip uninstall -y ${PACKAGE}

# Create a Python virtual environment
VENV=VENV_JEDGCC
python -m venv $VENV
source $VENV/bin/activate

# Activate this block to install from scratch
if [ 1 == 1 ]; then 
    python -m pip uninstall -y ${PACKAGE}
    SKBUILD=${PACKAGE_ROOT}/_skbuild;                    [ -d $SKBUILD ] && rm -r $SKBUILD
    EGGINFO=${PACKAGE_ROOT}/${PACKAGE}.egg-info;         [ -d $EGGINFO ] && rm -r $EGGINFO
    EGGINFO=${PACKAGE_ROOT}/python/${PACKAGE}.egg-info/; [ -d $EGGINFO ] && rm -r $EGGINFO
fi

# To get the -march
#gcc -march=native -Q --help=target | grep march

#export BIPP_CMAKE_ARGS="-DBLUEBILD_BUILD_TYPE=RelWithDebInfo "
#BIPP_GPU=OFF python -m pip install --user --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT}
#export BIPP_VC=ON
#-fopt-info-vec-missed
# -gdwarf-4
GCC_ASM="-S -masm=intel -fverbose-asm"
export BIPP_CMAKE_ARGS="-DBIPP_BUILD_TYPE=DEBUG   -DCMAKE_CXX_FLAGS_DEBUG=\"$GCC_ASM -g -m64 -O3 -fopenmp -ffast-math -march=icelake-server -mprefer-vector-width=512 -lm -fno-builtin-sincos\""
#export BIPP_CMAKE_ARGS="-DBIPP_BUILD_TYPE=DEBUG   -DCMAKE_CXX_FLAGS_DEBUG=\"         -g -m64 -O3 -fopenmp -ffast-math -march=icelake-server -mprefer-vector-width=512 -lm -fno-builtin-sincos\""
#export BIPP_CMAKE_ARGS="-DBIPP_BUILD_TYPE=RELEASE -DCMAKE_CXX_FLAGS_RELEASE=\"   -m64 -O3 -fopenmp -ffast-math -march=icelake-server -mprefer-vector-width=512 -lm -fno-builtin-sincos\""
export VERBOSE=1
BIPP_GPU=OFF python -m pip install --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT}

python -c "import ${PACKAGE}"
echo "-I- test [python -c \"import ${PACKAGE}\"] successful"

HOST_K_DIR=./bipp-jed/_skbuild/linux-x86_64-3.10/cmake-build/src/CMakeFiles/bipp.dir/host/kernels
echo $HOST_K_DIR
objdump -S -d ${HOST_K_DIR}/gemmexp.cpp.o > ${HOST_K_DIR}/gemmexp.dump
cat ${HOST_K_DIR}/gemmexp.dump

deactivate

source ~/SKA/ska-spack-env/bipp-jed-gcc/deactivate.sh

cd -

#[4/18] release mode
#/work/ska/soft/spack/blackhole/v1/opt/view_jed_gcc/bin/c++
#-Dbipp_EXPORTS -I/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/src
#-I/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/include
#-I/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/_skbuild/linux-x86_64-3.10/cmake-build
#-isystem /work/ska/soft/spack/blackhole/v1/opt/view_jed_gcc/include
#-O3 -DNDEBUG -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -MD -MT
#src/CMakeFiles/bipp.dir/host/kernels/gemmexp.cpp.o -MF src/CMakeFiles/bipp.dir/host/kernels/gemmexp.cpp.o.d -o src/CMakeFiles/bipp.dir/host/kernels/gemmexp.cpp.o -c /home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/src/host/kernels/gemmexp.cpp

#
#/work/ska/soft/spack/blackhole/v1/opt/view_jed_gcc/bin/c++
#-Dbipp_EXPORTS -I/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/src
#-I/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/include
#-I/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/_skbuild/linux-x86_64-3.10/cmake-build
#-isystem /work/ska/soft/spack/blackhole/v1/opt/view_jed_gcc/include
#-S -g -m64 -O3 -fopenmp -ffast-math -march=icelake-server -mprefer-vector-width=512 -lm -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -MD -MT src/CMakeFiles/bipp.dir/host/kernels/gemmexp.cpp.o -MF src/CMakeFiles/bipp.dir/host/kernels/gemmexp.cpp.o.d -o src/CMakeFiles/bipp.dir/host/kernels/gemmexp.cpp.o -c /home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-jed/src/host/kernels/gemmexp.cpp
