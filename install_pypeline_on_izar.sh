#!/bin/bash

set -e

# Old CPU SS needs Marla when compiled with GCC
export MARLA_ROOT=~/SKA/epfl-radio-astro/marla
export LD_LIBRARY_PATH=$FINUFFT_ROOT/lib:$CUFINUFFT_ROOT/lib:$umpire_DIR/lib:$LD_LIBRARY_PATH


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

CLUSTER=izar
PACKAGE=pypeline

PACKAGE_NAME="${PACKAGE}-${CLUSTER}"
PACKAGE_ROOT=${SCRIPT_DIR}/${PACKAGE_NAME}
echo PACKAGE_ROOT = ${PACKAGE_ROOT}

# Delete if exists already + Git clone 
# Install ci-master which does not contain bipp!
if [ 1 == 0 ]; then
    [ -d $PACKAGE_ROOT ] && rm -rf ${PACKAGE_ROOT}
    git clone --branch ci-master https://github.com/epfl-radio-astro/${PACKAGE}.git ${PACKAGE_ROOT}
fi

[ -d $PACKAGE_ROOT ] || (echo "-E- ${PACKAGE_ROOT} directory does not exist" && exit 1)

MY_SPACK_ENV=bipp-izar-gcc
source ~/SKA/ska-spack-env/${MY_SPACK_ENV}/activate.sh
python -V
which g++
g++ --version

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
