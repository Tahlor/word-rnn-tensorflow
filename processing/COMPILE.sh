#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32400M   # memory per CPU core
#SBATCH --gres=gpu:0
#SBATCH --output="COMPILE.slurm"

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module


module purge
module load defaultenv
module load cuda/8.0
module load cudnn/5.1_cuda-8.0
module load python/2/7


cd /fslhome/tarch/compute/673/word-rnn-tensorflow
cd ./processing
python ./5_compile.py

# To run:
#sbatch ./COMPILE.sh