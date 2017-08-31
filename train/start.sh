#!/bin/bash
DATASET=cuhk03
EXPNAME=partnet

if [ "$1" != "" ]
then
    DATASET=$1
fi

if [ "$2" != "" ]
then
    EXPNAME=$2
fi

python proto.py --dataset=${DATASET} --exp-name=${EXPNAME}

export PYTHONPATH=$(pwd)/../lib/python_layer:$PYTHONPATH

cd ${DATASET}/${EXPNAME}
chmod +x ./run*.sh
export GLOG_log_dir=./snapshot
$SHELL
#run_partnet.bat ${GPU_ID}

