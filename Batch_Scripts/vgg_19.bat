#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16GB
#SBATCH -p gpu
#SBATCH --gres="gpu:k40:1"
#SBATCH -J RNA-Vggnet19

model="vgg_19"

module load Python/3.5.2
module load slurm-torque/14.11.8
module load CUDA/9.0.176
module load cuDNN/7-CUDA-9.0.176

cd /nfs/apps/7/arch/generic/Python/3.5.2/bin
pyvenv /scratch/b523m844/python352/bin/
cd /scratch/b523m844/python352/bin/
source activate

rm /scratch/b523m844/RNA_Secondary_Structure_Classification/$model/train_loss.txt
rm /scratch/b523m844/RNA_Secondary_Structure_Classification/$model/val_loss.txt

python ~/RNA_Secondary_Structure_Classification/models/vgg_19.py

cp /scratch/b523m844/RNA_Secondary_Structure_Classification/$model/train_loss.txt ~/RNA_Secondary_Structure_Classification/utils/vgg_19_train_loss.txt
cp /scratch/b523m844/RNA_Secondary_Structure_Classification/$model/val_loss.txt ~/RNA_Secondary_Structure_Classification/utils/vgg_19_val_loss.txt