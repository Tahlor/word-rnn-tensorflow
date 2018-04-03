#!/bin/bash

#SBATCH --time=0:03:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=16400M   # memory per CPU core
#SBATCH --gres=gpu:0
#SBATCH --output="test.slurm"

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module


module purge
module load defaultenv
module load cuda/8.0
module load cudnn/5.1_cuda-8.0
module load python/2/7

python train.py --data_dir ./data/large --rnn_size 256 --num_layers 2 --model lstm --batch_size 256 --seq_length 32 --num_epochs 5000 --bonus True

# To run:
#sbatch ./train.sh