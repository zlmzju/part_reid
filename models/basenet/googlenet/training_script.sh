#!/bin/bash
# Name of the job: PBS can read the command after #, which shell ignores them
#PBS -N PROJECT_NAME

# Job asks for 1 node (computer) and 2 gpu on each of them (total 4 GPUs)
#PBS -l nodes=1:ppn=1

# Redirect standard error to standard output
#PBS -j oe

### Output files
#PBS -o /share/data/joblog/${PBS_JOBID}.OU

#Remap the free gpus to make the IDs always start from 0
source /share/data/script/util/remap_free_gpus.sh

# Enter the job's working directory.
[ "$PBS_O_WORKDIR" != "" ] && cd $PBS_O_WORKDIR

# Our own script
export GLOG_log_dir=./snapshot
PROJECT_DIR/caffe/build/tools/caffe train -solver solver.prototxt -weights PROJECT_DIR/models/basenet/googlenet/bvlc_googlenet.caffemodel -gpu 0
