#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH -p gpu
#SBATCH --gres="gpu:k40:1"
#SBATCH -J RNA-Resnet50

module load Python/3.5.2
module load slurm-torque/14.11.8
module load CUDA/9.0.176
module load cuDNN/7-CUDA-9.0.176

cd /nfs/apps/7/arch/generic/Python/3.5.2/bin
pyvenv /scratch/b523m844/python352/bin/
cd /scratch/b523m844/python352/bin/
source activate

sshfs b523m844@deadpool.ittc.ku.edu:/data/mount_data /scratch/b523m844/RNA_Secondary_Structure_Classification/data

python ~/RNA_Secondary_Structure_Classification/Resnet_50/resnet.py

cd /scratch/b523m844/RNA_Secondary_Structure_Classification
fusermount -u data