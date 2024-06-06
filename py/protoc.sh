#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]}) && pwd)

echoExec source ~/miniconda3/bin/activate grpc

if ls ./*_pb2*.py &>/dev/null; then
    echoExec rm ./*_pb2*.py
fi

echoExec python -m grpc_tools.protoc \
    --python_out=. --grpc_python_out=. -I$(cd .. && pwd)/src \
    $(cd .. && pwd)/src/service.proto
