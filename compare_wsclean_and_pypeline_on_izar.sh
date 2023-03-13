#!/bin/bash

set -e

MY_SPACK_ENV=bipp-izar-gcc

source /home/orliac/SKA/ska-spack-env/${MY_SPACK_ENV}/activate.sh
python -V

source ./VENV_IZARGCC/bin/activate
#python -m pip list

#python -m pip install python-casacore

WSC_SIZE=1000
WSC_SCALE=10

echo WSC_SIZE  = ${WSC_SIZE}
echo WSC_SCALE = ${WSC_SCALE}

if [ 1 == 0 ]; then
    echo =============================================================================
    echo python lofar_bootes_ss_pypeline.py
    echo =============================================================================
    python lofar_bootes_ss_pypeline.py --outdir . \
        --cluster izar --processing_unit gpu --compiler cuda --precision double \
        --package pypeline --nlev 4 \
        --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
        --sigma=1.0 --time_slice_pe 100 --time_slice_im 20
fi 

MS_FILE=gauss4_t201806301100_SBL180.MS

if [ 1 == 0 ]; then
    echo =============================================================================
    echo WSClean
    echo =============================================================================
   
    #time chgcentre -flipuvwsign $MS_FILE

    #-nwlayers 4 \
    #-make-psf \
    #-padding 1.0 \

    time wsclean \
        -verbose \
        -log-time \
        -channel-range 0 1 \
        -size ${WSC_SIZE} ${WSC_SIZE} \
        -scale ${WSC_SCALE}asec \
        -pol I \
        -weight natural \
        -niter 0 \
        -name test02 \
        ${MS_FILE} \
        | tee ${MS_FILE}_wsclean.log

fi

if [ 1 == 1 ]; then
    echo =============================================================================
    echo python real_lofar_bootes_ss_pypeline.py
    echo =============================================================================

    time python real_lofar_bootes_ss_pypeline.py \
        --ms_file ${MS_FILE} \
        --outdir . \
        --cluster izar \
        --processing_unit gpu --compiler cuda \
        --precision double \
        --package pypeline --nlev 4 \
        --pixw ${WSC_SIZE} --wsc_scale ${WSC_SCALE} \
        --sigma=0.9999 --time_slice_pe 100 --time_slice_im 100 \
        --wsc_log ${MS_FILE}_wsclean.log

fi



deactivate

source /home/orliac/SKA/ska-spack-env/${MY_SPACK_ENV}/deactivate.sh
