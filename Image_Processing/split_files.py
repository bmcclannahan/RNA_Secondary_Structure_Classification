import os, glob
from random import random

train = .7
val = .2
test = .1
source_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/"
dest_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/Big_Training_Set/"
files = glob.glob(source_dir+"*.jpg")
print("Number of files:", len(files))

for f in files:
    split = f.split('/')
    second_split = split[4].split('_')
    folder = second_split[0]
    family = "Diff_Family"
    if second_split[-1] == '1.jpg':
        family = "Same_Family"
    new_dir = dest_dir+folder+"/"+family+"/"
    file_name = f.split('/')[-1]
    new_file = new_dir+file_name
    os.rename(f,new_file)