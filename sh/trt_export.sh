#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

echoExec trtexec --onnx=static/FaceDetector.onnx \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:1x3x640x640 \
    --maxShapes=input:1x3x640x640 \
    --saveEngine=static/FaceDetector.engine
