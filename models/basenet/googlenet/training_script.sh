#!/bin/bash
# Our own script
export GLOG_log_dir=./snapshot
PROJECT_DIR/caffe/build/tools/caffe train -solver solver.prototxt -weights PROJECT_DIR/models/basenet/googlenet/bvlc_googlenet.caffemodel -gpu 0
