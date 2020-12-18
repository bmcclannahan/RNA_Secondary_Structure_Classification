from glob import glob
from random import randint
import os

file_dir = "/data/rna_classification/val/Diff_Family/"
current_dir = "/home/b523m844/data/rna_classification/val/Diff_Family"
files = glob(current_dir+"*.jpg")

old_files = glob(file_dir+*".jpg")

for f in files:
    file_name = f.split['/'][-1]
    old_name = file_dir+file_name
    if old_name in old_files:
        #os.replace(f,old_name)
        print(old_name)
        print(f)