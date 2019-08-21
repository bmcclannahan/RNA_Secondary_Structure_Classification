#!/bin/bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2GB
#SBATCH -J RNA_Training_Set_Creation

module load MATLAB/2019a
cd /users/b523m844/RNA_Secondary_Structure_Classification/Image_Processing
cp -rf *.m /scratch/b523m844/RNA_Secondary_Structure_Classification
cd /scratch/b523m844/RNA_Secondary_Structure_Classification
matlab -nodisplay -nosplash -nodesktop -r "run('/scratch/b523m844/RNA_Secondary_Structure_Classification/generate_data_set.m');exit;"
python3 /users/b523m844/RNA_Secondary_Structure_Classification/Image_Processing/split_files.py
