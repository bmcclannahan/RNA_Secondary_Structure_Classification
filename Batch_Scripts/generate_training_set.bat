#!/bin/bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH --mem=16GB
#SBATCH -J RNA_Training_Set_Creation

module load MATLAB/2019a
sshfs b523m844@deadpool.ittc.ku.edu:/data/mount_data /scratch/b523m844/RNA_Secondary_Structure_Classification/data
cd /users/b523m844/RNA_Secondary_Structure_Classification/Image_Processing
cp -rf *.m /home/b523m844/data/rna_classification/val
cd /home/b523m844/data/rna_classification/val
matlab -nodisplay -nosplash -nodesktop -r "run ('/home/b523m844/data/rna_classification/generate_data_set.m');exit;"
