#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16GB
#SBATCH -p gpu
#SBATCH --gres="gpu:k80:1"
#SBATCH -J RNA-ROC
#SBATCH -t 48:00:00

model="siamese_v19"

module load Python/3.5.2
module load slurm-torque/14.11.8
module load CUDA/9.0.176
module load cuDNN/7-CUDA-9.0.176

cd /nfs/apps/7/arch/generic/Python/3.5.2/bin
pyvenv /scratch/b523m844/python352/bin/
cd /scratch/b523m844/python352/bin/
source activate

python ~/RNA_Secondary_Structure_Classification/models/generate_roc_curves.py