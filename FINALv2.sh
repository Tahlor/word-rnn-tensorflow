#!/bin/bash

#SBATCH --time=36:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32400M   # memory per CPU core
#SBATCH --gres=gpu:4
#SBATCH --output="FINALv2.slurm"

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#%Module


module purge
module load defaultenv
module load cuda/8.0
module load cudnn/5.1_cuda-8.0
module load python/2/7

cd /fslhome/tarch/compute/673/word-rnn-tensorflow2
python -u train.py --data_dir ./data/FINALv2 --rnn_size 128 --num_layers 2 --model gru --batch_size 256 --seq_length 50 --num_epochs 5000 --bonus True --sample False  --save_dir "./save/FINALv2" --end_word_training True --init_from "./save/FINALv2"

# To run:
#sbatch ./FINALv2.sh