#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]})/../src && pwd)

if ls ./*.pb.* &>/dev/null; then
    echoExec rm ./*.pb.*
fi

echoExec protoc \
    --grpc_out=. --cpp_out=. -I. \
    --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
    service.proto
