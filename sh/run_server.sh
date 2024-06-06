#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

if [ -f bin/server ]; then
    echoExec ./bin/server localhost:50051 static/FaceDetector.engine
fi
