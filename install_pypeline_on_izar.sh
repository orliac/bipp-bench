#!/bin/bash

# Install ci-master which does not contain bipp!

export MARLA_ROOT=~/SKA/epfl-radio-astro/marla

export LD_LIBRARY_PATH=$FINUFFT_ROOT/lib:$CUFINUFFT_ROOT/lib:$umpire_DIR/lib:$LD_LIBRARY_PATH
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

PYPELINE_ROOT=${SCRIPT_DIR}/pypeline-izar
echo PYPELINE_ROOT = $PYPELINE_ROOT

#[ -d ${PYPELINE_ROOT} ] && rm -rf ${PYPELINE_ROOT}
#git clone --branch ci-master https://github.com/epfl-radio-astro/pypeline.git pypeline-izar
[ -d ${PYPELINE_ROOT} ] || (echo "-E- pypeline root directory does not exist" && exit 1)

source ~/SKA/ska-spack-env/env-bipp-izar/activate.sh
which python
python -V

# Install bipp from scratch
# 
if [ 1 == 1 ]; then 
    #SKBUILD=${PYPELINE_ROOT}/_skbuild;              [ -d $SKBUILD ] && rm -r $SKBUILD
    #EGGINFO=${PYPELINE_ROOT}/bipp.egg-info;         [ -d $EGGINFO ] && rm -r $EGGINFO
    #EGGINFO=${PYPELINE_ROOT}/python/bipp.egg-info/; [ -d $EGGINFO ] && rm -r $EGGINFO
    #python3 -m pip list 
    python3 -m pip uninstall -y bluebild
    python3 -m pip uninstall -y pypeline
    #python3 -m pip list
    #export BIPP_CMAKE_ARGS="-DBLUEBILD_BUILD_TYPE=RelWithDebInfo "
    python3 -m pip install --verbose --user --no-deps --no-build-isolation "${PYPELINE_ROOT}/src/bluebild"
    python3 -m pip install           --user --no-deps --no-build-isolation "${PYPELINE_ROOT}"
fi

source ~/SKA/ska-spack-env/env-bipp-izar/deactivate.sh

cd -
