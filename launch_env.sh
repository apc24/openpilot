#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MAPTILER_TOKEN=""
export MAPBOX_TOKEN=""

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="9.6"
fi

export STAGING_ROOT="/data/safe_staging"
