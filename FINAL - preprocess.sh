#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32400M   # memory per CPU core
#SBATCH --gres=gpu:4
#SBATCH --output="FINAL-process.slurm"

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module


module purge
module load python/3/6

#cd /fslhome/tarch/compute/673/word-rnn-tensorflow
#cd ./processing
#python3 -u ./5_compile.py
#mv "../data/FINAL/gutenberg2_restricted.txtNEW" "../data/FINAL/input.txt"

cd /fslhome/tarch/compute/673/word-rnn-tensorflow
python3 -u utils.py --path "./data/FINAL"

# To run:
#sbatch ./FINAL.sh