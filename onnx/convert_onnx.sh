#!/bin/bash

MEL_FILTERS_PATH=$1
CHUNK_LENGTH=$2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/.. python $SCRIPT_DIR/convert_onnx.py --mel_filters_path $MEL_FILTERS_PATH --chunk_length $CHUNK_LENGTH 2>&1 | grep -v -e Ignore -e warning -e Warning -e return
