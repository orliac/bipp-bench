#!/bin/bash

set -e

# Old CPU SS needs Marla when compiled with GCC
export MARLA_ROOT=~/SKA/epfl-radio-astro/marla
export LD_LIBRARY_PATH=$FINUFFT_ROOT/lib:$CUFINUFFT_ROOT/lib:$umpire_DIR/lib:$LD_LIBRARY_PATH


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
PACKAGE=pypeline

PACKAGE_NAME="${PACKAGE}-${CLUSTER}-${github_workspace}"
PACKAGE_ROOT=${SCRIPT_DIR}/${PACKAGE_NAME}
echo PACKAGE_ROOT = ${PACKAGE_ROOT}


# Delete if exists already + Git clone
# /!\ Install ci-master which does not contain bipp!
#
if [ $from_scratch == 1 ]; then
    echo "-W- ARE YOU SURE ABOUT THE RECURSIVE DELETE? If so comment me out." && exit 1
    [ -d $PACKAGE_ROOT ] && rm -rf ${PACKAGE_ROOT}
    git clone --branch ci-master https://github.com/${github_workspace}/${PACKAGE}.git ${PACKAGE_ROOT}
fi

# Git clone branch ci-master if not done yet
if [ ! -d $PACKAGE_ROOT ]; then 
    echo "-W- ${PACKAGE_ROOT} directory does not exist, will clone it"
    git clone --branch ci-master https://github.com/${github_workspace}/${PACKAGE}.git ${PACKAGE_ROOT}
fi

# Do not pull automatically!

# Activate Spack environment for izar
MY_SPACK_ENV=bipp-izar-gcc
source ~/SKA/ska-spack-env/${MY_SPACK_ENV}/activate.sh
python -V

# Just in case, remove pypeline & bluebild from outside any virtual env
python -m pip uninstall -y bluebild
python -m pip uninstall -y pypeline

# Create a Python virtual environment
VENV=VENV_IZARGCC
python -m venv $VENV
source $VENV/bin/activate

# Activate this block to install from scratch
if [ 1 == 0 ]; then 
    python -m pip uninstall -y bluebild
    python -m pip uninstall -y pypeline
    SKBUILD=${PACKAGE_ROOT}/src/bluebild/_skbuild;
    [ -d $SKBUILD ] && (echo "-W- removing $SKBUILD" && rm -r $SKBUILD)
    EGGINFO=${PACKAGE_ROOT}/${PACKAGE}.egg-info;         [ -d $EGGINFO ] && rm -r $EGGINFO
    EGGINFO=${PACKAGE_ROOT}/python/${PACKAGE}.egg-info/; [ -d $EGGINFO ] && rm -r $EGGINFO
fi

#export BLUEBILD_CMAKE_ARGS="-DBLUEBILD_BUILD_TYPE=RelWithDebInfo "
#BLUEBILD_GPU=OFF python -m pip install --user --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT}
#export CUDAFLAGS="-g -G"
BLUEBILD_GPU=CUDA python -m pip install --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT}/src/bluebild
#python -m pip install --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT}
python -m pip install --verbose --no-deps -e ${PACKAGE_ROOT}

python -c "import pypeline"
echo "-I- test [python -c \"import ${PACKAGE}\"] successful"
python -c "import bluebild"
echo "-I- test [python -c \"import bluebild\"] successful"

deactivate

source ~/SKA/ska-spack-env/${MY_SPACK_ENV}/deactivate.sh

cd -
