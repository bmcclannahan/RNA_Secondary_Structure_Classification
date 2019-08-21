import os, glob
from random import random

train = .7
val = .2
test = .1
source_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/Big_Training_Set/"
dest_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/Big_Training_Set/"

for file in glob.glob(source_dir+"*.jpg"):
    num = random()
    folder = ''
    if num < val:
        folder = 'val/'
    else if (1-num) < test:
        folder = 'test/'
    else:
        folder = 'train/'
    new_dir = dest_dir+folder
    file_name = file.split('/')[-1]