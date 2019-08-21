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
    num = random()
    folder = ''
    if num < val:
        folder = 'val/'
    elif (1-num) < test:
        folder = 'test/'
    else:
        folder = 'train/'
    new_dir = dest_dir+folder
    file_name = f.split('/')[-1]
    new_file = new_dir+file_name
    os.rename(f,new_file)