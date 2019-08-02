#!/bin/bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH -p gpu
#SBATCH --gres="gpu:k20:1"
#SBATCH -J Python-Test

module load Python/3.5.2
pip3 install --user torch
pip3 install --user torchvision
pip3 install --user matplotlib

python3 test.py