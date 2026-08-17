#!/bin/bash
# Download the CAPI-WOMEN datasets (UMD, UT-EndoMRI) into $SHAPEDEM_ROOT/data.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${SHAPEDEM_ROOT:-$REPO/_workspace}"
DATA="$ROOT/data"; mkdir -p "$DATA"
PY="${PY:-python}"
curl -L -C - --retry 8 --retry-all-errors -o "$DATA/UMD.zip" \
  "https://ndownloader.figshare.com/files/44111183"
curl -L -C - --retry 8 --retry-all-errors -o "$DATA/UT-EndoMRI.zip" \
  "https://zenodo.org/api/records/13749613/files/UT-EndoMRI.zip/content"
ls -lh "$DATA/UMD.zip" "$DATA/UT-EndoMRI.zip"
echo "CAPI_DL_DONE"
