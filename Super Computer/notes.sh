module load defaultenv
module load cuda/8.0
module load cudnn/5.1_cuda-8.0
module load python/3/6

# For scikit etc.:
module load anaconda/3/4.3.1


pip install --user package
salloc --mem 16000M --time 2:00:00 --gres=gpu:2
