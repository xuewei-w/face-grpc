#!/bin/bash
echoExec() {
    echo $*
    $*
}
echoExec cd $(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

if [ -d ./build ]; then
    echoExec rm -r ./build
fi

echoExec mkdir build
echoExec cd build
echoExec cmake ..
echoExec make
echoExec cd ..
echoExec rm -r build
