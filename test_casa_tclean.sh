#!/bin/bash

set -

CASA=~/SKA/casa-6.5.3-28-py3.8/bin/casa
[ -f $CASA ] || (echo "Fatal. Could not find $CASA" && exit 1)
which $CASA

#CASAVIEWER=~/SKA/casa-6.5.3-28-py3.8/bin/casaviewer
#[ -f $CASAVIEWER ] || (echo "Fatal. Could not find $CASAVIEWER" && exit 1)

#$CASA --help

$CASA \
    --notelemetry \
    --logfile casa.log \
    -c casa_tclean.py

#$CASAVIEWER casa_dirty.image
