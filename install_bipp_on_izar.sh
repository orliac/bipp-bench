#!/bin/bash

set -e

# Check command line input (getopt not ideal as --f* will enable --from_scratch)
#
if [[ $# < 1 ]] || [[ $# > 2 ]]; then
    echo "-E- At least one arguement is expected, the name of the repo to be used [upstream, orliac]"
    echo "-E- Max 2 arguments are expected, optional second one is --from-scratch"
    exit 1
fi
from_scratch=0
if [[ $# == 2 ]]; then
    if [ $2 != "--from_scratch" ]; then
        echo "-E- If second arg is passed, must be --from_scratch"
        exit 1
    else
        from_scratch=1
    fi
fi
echo "-I- from_scratch = $from_scratch"
github_workspace=$1
if [ $github_workspace !=  "epfl-radio-astro" ] && [ $github_workspace !=  "orliac" ]; then
    echo "-E- Unknown GitHub workspacke $workspace. Must be \"epfl-radio-astro\" or \"orliac\"."
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

CLUSTER=izar
PACKAGE=bipp
PACKAGE_NAME="${PACKAGE}-${CLUSTER}-${github_workspace}"
PACKAGE_ROOT=${SCRIPT_DIR}/${PACKAGE_NAME}
echo PACKAGE_ROOT = ${PACKAGE_ROOT}

# Delete if exists already + Git clone
#
if [ $from_scratch == 1 ]; then
    echo "-W- ARE YOU SURE ABOUT THE RECURSIVE DELETE? If so comment me out." && exit 1
    [ -d $PACKAGE_ROOT ] && rm -rf ${PACKAGE_ROOT}
    git clone https://github.com/${github_workspace}/${PACKAGE}.git ${PACKAGE_ROOT}
fi

# Git clone if not done yet 
if [ ! -d $PACKAGE_ROOT ]; then 
    echo "-W- ${PACKAGE_ROOT} directory does not exist, will clone it"
    git clone https://github.com/${github_workspace}/${PACKAGE}.git ${PACKAGE_ROOT}
fi

cd ${PACKAGE_ROOT}
git branch
cd -

# Do not pull automatically!


# Activate Spack environment for izar
MY_SPACK_ENV=bipp-izar-gcc
source ~/SKA/ska-spack-env/${MY_SPACK_ENV}/activate.sh
python -V

# Just in case, remove bipp from outside any virtual env
python -m pip uninstall -y ${PACKAGE}

# Create a Python virtual environment
VENV=VENV_IZARGCC
python -m venv $VENV
source $VENV/bin/activate

# Activate this block to install from scratch
if [ 1 == 0 ]; then 
    python -m pip uninstall -y ${PACKAGE}
    ls ${PACKAGE_ROOT}
    SKBUILD=${PACKAGE_ROOT}/_skbuild;
    [ -d $SKBUILD ] && (echo "-I- removing $SKBUILD" && rm -r $SKBUILD)
    EGGINFO=${PACKAGE_ROOT}/${PACKAGE}.egg-info;
    [ -d $EGGINFO ] && (echo "-I- removing $EGGINFO" && rm -r $EGGINFO)
    EGGINFO=${PACKAGE_ROOT}/python/${PACKAGE}.egg-info/;
    [ -d $EGGINFO ] && (echo "-I- removing $EGGINFO" && rm -r $EGGINFO)
fi

#export BIPP_VC=ON

# cuda_arch="60;61;70;75;80;86"
# -DBIPP_GPU=CUDA
# -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch}
# -DCMAKE_CXX_FLAGS="-march=${cpu_arch}"
# -DCMAKE_C_FLAGS="-march=${cpu_arch}"
# -DCMAKE_CUDA_FLAGS="--compiler-options=\"-march=${cpu_arch}\""

#export BIPP_CMAKE_ARGS="-DBIPP_BUILD_TYPE=Debug"
cuda_arch="70;80"
cpu_arch="cascadelake"

production_mode=1

if [[ $production_mode == 1 ]]; then
    echo "@@@ compiling BIPP in production mode @@@"
    export BIPP_CMAKE_ARGS="-DBIPP_BUILD_TYPE=Release \
                            -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch}"
else
    echo "@@@ compiling BIPP in debug mode @@@"
    export BIPP_CMAKE_ARGS="-DBIPP_BUILD_TYPE=Debug \
                            -DCMAKE_CXX_FLAGS_DEBUG=\"-g -O3 -march=${cpu_arch} -mprefer-vector-width=512 -ftree-vectorize\" \
                            -DCMAKE_C_FLAGS_DEBUG=\"-g -O3 -march=${cpu_arch} -mprefer-vector-width=512 -ftree-vectorize\" \
                            -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} \
                            -DCMAKE_CUDA_FLAGS_DEBUG=\"-g -lineinfo --compiler-options=\"-march=${cpu_arch}\"\""
fi

echo "BIPP_CMAKE_ARGS: $BIPP_CMAKE_ARGS"

#export CUDAFLAGS="-g -G"
# Outside a venv
#BIPP_GPU=CUDA python -m pip install --user --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT} # For prod
#BIPP_GPU=CUDA python -m pip install --user --verbose --no-deps -e ${PACKAGE_ROOT} # For dev
# Inside a venv
BIPP_GPU=CUDA python -m pip install --no-deps --no-build-isolation ${PACKAGE_ROOT} # For prod
#BIPP_GPU=CUDA python -m pip install --verbose --no-deps -e ${PACKAGE_ROOT} # For dev

python -m pip list | grep bipp

python -c "import ${PACKAGE}"
echo "-I- test [python -c \"import ${PACKAGE}\"] successful"

deactivate

source ~/SKA/ska-spack-env/${MY_SPACK_ENV}/deactivate.sh
cd -
