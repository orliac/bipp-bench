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

#export BIPP_CMAKE_ARGS="-DBLUEBILD_BUILD_TYPE=RelWithDebInfo "
#BIPP_GPU=OFF python -m pip install --user --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT}
BIPP_GPU=OFF python -m pip install --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT}

python -c "import ${PACKAGE}"
echo "-I- test [python -c \"import ${PACKAGE}\"] successful"

deactivate

source ~/SKA/ska-spack-env/bipp-jed-gcc/deactivate.sh

cd -
