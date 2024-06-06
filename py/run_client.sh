#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]}) && pwd)

echoExec source ~/miniconda3/bin/activate grpc

if [ -f ./output.jpg ]; then
    echoExec rm output.jpg
fi

echoExec python client.py \
    localhost:50051 $(cd .. && pwd)/static/test.jpg output.jpg

if [ -d ./__pycache__ ]; then
    echoExec rm -r ./__pycache__
fi
