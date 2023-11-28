#!/bin/bash

set -e

### Screening command line arguments
args=$(getopt -a -o d:r: --long dsk,repo: -- "$@")
if [[ $? -gt 0 ]]; then
  usage
fi

DSK=0
eval set -- ${args}
while :
do
  case $1 in
    -d | --dsk)    DSK=1    ; shift   ;;
    -r | --repo)   REPO=$2   ; shift 2 ;;
    --) shift; break ;;
    *) >&2 echo Unsupported option: $1
       usage ;;
  esac
done


echo "-I- Githup workspace to be used for bipp: $REPO"
echo "-I- Delete _skbuild before installing BIPP? $DSK"

echo "-D- Remainging/ignored parameters from call to $0 are: $@"


# Check command line input (getopt not ideal as --f* will enable --from_scratch)
#
github_workspace=${REPO}
if [ $github_workspace !=  "epfl-radio-astro" ] && [ $github_workspace !=  "orliac" ] && \
       [ $github_workspace !=  "arpan" ]; then
    echo "-E- Unknown GitHub workspacke $workspace. Must be \"epfl-radio-astro\" or \"orliac\" or \"arpan\"."
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

CLUSTER=izar
PACKAGE=bipp
PACKAGE_NAME="${PACKAGE}-${CLUSTER}-${github_workspace}"
PACKAGE_ROOT=${SCRIPT_DIR}/${PACKAGE_NAME}
echo PACKAGE_ROOT = ${PACKAGE_ROOT}

cd ${PACKAGE_ROOT}
git branch
cd -

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

if [[ $DSK == 1 ]]; then
    SKBUILD=${PACKAGE_ROOT}/_skbuild;
    if [ -d ${SKBUILD} ]; then
        echo "-W- About to delete ${SKBUILD} (in 5 sec :-))"
        sleep 5
        rm -r ${SKBUILD}
    else
        echo "-W- ${SKBUILD} not found."
    fi
fi

# Activate this block to install from scratch
if [ 1 == 0 ]; then 
    ls ${PACKAGE_ROOT}
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

export 

cuda_arch="70;80"
cpu_arch="cascadelake"

production_mode=1

if [[ $production_mode == 1 ]]; then
    echo "@@@ compiling BIPP in production mode @@@"
    #BIPP_CMAKE_ARGS+=" -DCMAKE_CXX_FLAGS_RELEASE=\"-Ofast -march=${cpu_arch} -mprefer-vector-width=512 -ftree-vectorize\""
    #BIPP_CMAKE_ARGS+=" -DCMAKE_C_FLAGS_RELEASE=\"-Ofast -march=${cpu_arch} -mprefer-vector-width=512 -ftree-vectorize\""
    BIPP_CMAKE_ARGS="-DCMAKE_CXX_FLAGS_RELEASE=\"-march=${cpu_arch} -Ofast -DNDEBUG\""
    BIPP_CMAKE_ARGS+=" -DCMAKE_C_FLAGS_RELEASE=\"-march=${cpu_arch} -Ofast -DNDEBUG\""
    BIPP_CMAKE_ARGS+=" -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DBIPP_BUILD_TESTS=ON"
    BIPP_CMAKE_ARGS+=" -DCMAKE_CUDA_FLAGS_RELEASE=\"--compiler-options=-march=${cpu_arch},-Ofast,-DNDEBUG\""
    echo $BIPP_CMAKE_ARGS
    export BIPP_CMAKE_ARGS=${BIPP_CMAKE_ARGS}
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
BIPP_GPU=CUDA python -m pip install --verbose --no-deps --no-build-isolation ${PACKAGE_ROOT} # For prod
#BIPP_GPU=CUDA python -m pip install --verbose --no-deps -e ${PACKAGE_ROOT} # For dev

python -m pip list | grep bipp

python -c "import ${PACKAGE}"
echo "-I- test [python -c \"import ${PACKAGE}\"] successful"

deactivate

source ~/SKA/ska-spack-env/${MY_SPACK_ENV}/deactivate.sh
cd -
