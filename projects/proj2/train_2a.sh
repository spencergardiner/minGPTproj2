#!/bin/bash --login

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=spencer.gardiner@gmail.com
#SBATCH --qos=dw87
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --job-name=2a_train



module purge
module load cuda

mamba activate kat

python ./proj2a.py