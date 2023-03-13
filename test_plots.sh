#!/bin/bash

module load gcc
module load python

set -e

OUT=/work/ska/orliac/benchmarks/paper01/bipp/izar/gpu/cuda/double/ss/0/4/1024/
[ -d $OUT ] || (echo "-E- $OUT not found" && exit 1)

python plots.py \
    --bb_grid   $OUT/I_lsq_eq_grid.npy \
    --bb_data   $OUT/I_lsq_eq_data.npy \
    --bb_json   $OUT/stats.json \
    --wsc_fits  $OUT/dirty_wsclean-dirty.fits \
    --wsc_log   $OUT/wsclean.log \
    --casa_fits $OUT/dirty_casa.image.fits \
    --casa_log  $OUT/casa.log

