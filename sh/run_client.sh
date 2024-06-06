#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

if [ -d ./output ]; then
    echoExec rm -r ./output
fi

if [ -f bin/client ]; then
    mkdir output
    echoExec ./bin/client localhost:50051 static/test.jpg output/output.jpg
fi
