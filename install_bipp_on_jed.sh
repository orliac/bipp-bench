#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

BIPP_NAME=bipp-jed

BIPP_ROOT=${SCRIPT_DIR}/${BIPP_NAME}
echo BIPP_ROOT = $BIPP_ROOT

# Delete + Git clone 
#[ -d $BIPP_ROOT ] && rm -rf ${BIPP_ROOT}
#git clone https://github.com/epfl-radio-astro/bipp.git ${BIPP_NAME}

[ -d $BIPP_ROOT ] || (echo "-E- bipp root directory does not exist" && exit 1)

# Activate Spack environment for izar
source ~/SKA/ska-spack-env/bipp-jed-gcc/activate.sh
python -V

# Just in case, remove bipp from outside any virtual env
python3 -m pip uninstall -y bipp

# Create a Python virtual environment for izar
# TODO(eo): check whether that allows to install and run simultaneously from jed and izar
VENV=VENV_BIPPJEDGCC
python -m venv $VENV
source $VENV/bin/activate

# Activate this block to install bipp from scratch
if [ 1 == 1 ]; then 
    python3 -m pip uninstall -y bipp
    SKBUILD=${BIPP_ROOT}/_skbuild;              [ -d $SKBUILD ] && rm -r $SKBUILD
    EGGINFO=${BIPP_ROOT}/bipp.egg-info;         [ -d $EGGINFO ] && rm -r $EGGINFO
    EGGINFO=${BIPP_ROOT}/python/bipp.egg-info/; [ -d $EGGINFO ] && rm -r $EGGINFO
fi

#export BIPP_CMAKE_ARGS="-DBLUEBILD_BUILD_TYPE=RelWithDebInfo "
#BIPP_GPU=OFF python3 -m pip install --user --verbose --no-deps --no-build-isolation ${BIPP_ROOT}
BIPP_GPU=OFF python3 -m pip install --verbose --no-deps --no-build-isolation ${BIPP_ROOT}

python -c "import bipp"
echo "-I- test [python -c \"import bipp\"] successful"

deactivate

source ~/SKA/ska-spack-env/bipp-jed-gcc/deactivate.sh

cd -
