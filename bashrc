#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export KCODEPATH=$DIR
export PYTHONPATH=$PYTHONPATH:$KCODEPATH/lib
export PATH=$PATH:$KCODEPATH/bin
