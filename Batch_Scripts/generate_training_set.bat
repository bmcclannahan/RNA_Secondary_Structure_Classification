#!/bin/bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --mem=2GB
#SBATCH -J test_job

module load MATLAB/2019a
matlab -nodisplay -nosplash -nodesktop -r "run('/users/b523m844/RNA_Secondary_Structure_Classification/generate_data_set.m');exit;"
