#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]}) && pwd)

echoExec source ~/miniconda3/bin/activate retinaface

echoExec python export.py \
    static/Resnet50_Final.pth \
    static/FaceDetector.onnx

if [ -d ./__pycache__ ]; then
    echoExec rm -r ./__pycache__
fi
