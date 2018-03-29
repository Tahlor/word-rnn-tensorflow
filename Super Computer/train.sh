#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32400M   # memory per CPU core
#SBATCH --gres=gpu:0
#SBATCH --output="data.slurm"

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module

#module purge
#module load gdb/7.9.1                                           
#module load compiler_gnu/4.9.2     
#module load mkl/11.1.2                             
#module load compiler_intel/14.0.2         
#module load compiler_pgi/14.1
#module load mpi/openmpi-1.8.4_gnu-4.9.2

module load defaultenv
#module load cuda/8.0
#module load cudnn/5.1_cuda-8.0
module load python/3/6

# For scikit etc.:
#module load anaconda/3/4.3.1

python3 ./process_data/trade_data.py

# To run:
#sbatch ./train.sh