#!/bin/bash

MEL_FILTERS_PATH=$1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR
PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/.. python $SCRIPT_DIR/convert_onnx.py --mel_filters_path $MEL_FILTERS_PATH
