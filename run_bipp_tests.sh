#!/bin/bash

set -e
echo "-I- running $0"

INSTALL_BIPP=0
while getopts b flag
do
    case "${flag}" in
        b) INSTALL_BIPP=1;;
        ?) exit 1
    esac
done

if [[ $INSTALL_BIPP == 1 ]]; then
    sh generate_bipp_json_test_files.sh -b -s
    sh generate_bipp_json_test_files.sh -d
else
    sh generate_bipp_json_test_files.sh -s
    sh generate_bipp_json_test_files.sh -d
fi


INDIR_SP="/work/ska/orliac/bipp_json_test_files/128/4/0-2-1-1-single"
INDIR_DP="/work/ska/orliac/bipp_json_test_files/128/4/0-2-1-1-double"

DATA_DIR=/home/orliac/SKA/epfl-radio-astro/bipp-bench/bipp-izar-orliac/tests/data
TELESCOPE="skalow"
TELESCOPE="lofar"

# Global -- copy input files from nufft
#        -- need to run a test_michele_real_lofar.sh with nufft and ss to generate output (either with CPU or GPU)
if [[ 1 == 1 ]]; then
    cp -v ${INDIR_SP}/${TELESCOPE}_nufft_input_bipp_single.json   ${DATA_DIR}/lofar_input.json
    cp -v ${INDIR_SP}/${TELESCOPE}_nufft_output_bipp_single.json  ${DATA_DIR}/lofar_nufft_output_single.json
    cp -v ${INDIR_DP}/${TELESCOPE}_nufft_output_bipp_double.json  ${DATA_DIR}/lofar_nufft_output_double.json
    cp -v ${INDIR_SP}/${TELESCOPE}_ss_output_bipp_single.json     ${DATA_DIR}/lofar_ss_output_single.json
    cp -v ${INDIR_DP}/${TELESCOPE}_ss_output_bipp_double.json     ${DATA_DIR}/lofar_ss_output_double.json
    ./bipp-izar-orliac/_skbuild/linux-x86_64-3.10/cmake-build/tests/run_tests
fi

# Standard Synthesis only
if [[ 1 == 0 ]]; then
    cp -v lofar_ss_input_bipp.json          ${DATA_DIR}/lofar_input.json
    cp -v lofar_ss_output_bipp_single.json  ${DATA_DIR}/lofar_ss_output_single.json
    cp -v lofar_ss_output_bipp_double.json  ${DATA_DIR}/lofar_ss_output_double.json
    ./bipp-izar-orliac/_skbuild/linux-x86_64-3.10/cmake-build/tests/run_tests --gtest_filter=*StandardSynthesis*
fi

# NUFFT Synthesis only
if [[ 1 == 0 ]]; then
    cp -v lofar_nufft_input_bipp.json          ${DATA_DIR}/lofar_input.json
    cp -v lofar_nufft_output_bipp_single.json  ${DATA_DIR}/lofar_nufft_output_single.json
    cp -v lofar_nufft_output_bipp_double.json  ${DATA_DIR}/lofar_nufft_output_double.json
    #./bipp-izar-orliac/_skbuild/linux-x86_64-3.10/cmake-build/tests/run_tests --gtest_filter=*NufftSynthesisLofarDouble*
    ./bipp-izar-orliac/_skbuild/linux-x86_64-3.10/cmake-build/tests/run_tests --gtest_filter=*NufftSynthesisLofar*
fi

