#!/bin/sh
# Bundle dataset, code, results and figures into NanoJournal/NanoNEXT_XRD_package.zip
cd "$(dirname "$0")/../.." || exit 1
rm -f NanoNEXT_XRD_package.zip
zip -rq -X NanoNEXT_XRD_package.zip \
  exp_bandgap_dataset \
  paper_project/code paper_project/results paper_project/figures \
  paper_project/README.md paper_project/requirements.txt \
  -x '*/__pycache__/*' '*.DS_Store'
