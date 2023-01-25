#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

BIPP_ROOT=${SCRIPT_DIR}/bipp-izar
echo BIPP_ROOT = $BIPP_ROOT


[ -d $BIPP_ROOT ] && rm -rf ${BIPP_ROOT}
git clone https://github.com/epfl-radio-astro/bipp.git bipp-izar
[ -d $BIPP_ROOT ] || (echo "-E- bipp root directory does not exist" && exit 1)

source ~/SKA/ska-spack-env/env-bipp-izar/activate.sh

# Install bipp from scratch
# 
if [ 1 == 1 ]; then 
    SKBUILD=${BIPP_ROOT}/_skbuild;              [ -d $SKBUILD ] && rm -r $SKBUILD
    EGGINFO=${BIPP_ROOT}/bipp.egg-info;         [ -d $EGGINFO ] && rm -r $EGGINFO
    EGGINFO=${BIPP_ROOT}/python/bipp.egg-info/; [ -d $EGGINFO ] && rm -r $EGGINFO
    python3 -m pip uninstall -y bipp
    #export BIPP_CMAKE_ARGS="-DBLUEBILD_BUILD_TYPE=RelWithDebInfo "
    BIPP_GPU=CUDA python3 -m pip install --user --verbose --no-deps --no-build-isolation ${BIPP_ROOT}
fi

source ~/SKA/ska-spack-env/env-bipp-izar/deactivate.sh

cd -
