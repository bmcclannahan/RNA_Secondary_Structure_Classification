import os, glob
from random import random

seed_data = "/users/b523m844/RNA_Secondary_Structure_Classification/seed_info/Medium_families.txt"
source_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/Full_Image_Set/Medium_Families/"

train = .7
val = .1
test = .2

train_families = []
val_families = []
test_families = []

with open(seed_data) as f:
    medium_families = f.readlines()
medium_families = [x.rstrip() for x in medium_families]

num_med_families = len(medium_families)

for fam in medium_families:
    split = fam.split(",")
    family = int(split[0][2:])
    rand = random()
    if rand < train and len(train_families) < train*num_med_families:
        train_families.append(family)
    elif rand < train + val and len(val_families) < val*num_med_families:
        val_families.append(family)
    else:
        test_families.append(family)

print("Number of train families:",len(train_families))
print("Number of val families:",len(val_families))
print("Number of test families:",len(test_families))

files = glob.glob(source_dir+"*.jpg")

for f in files:
    split = f.split('/')
    family_name = split[-1][:7]
    family = int(family_name[2:])
    folder = ""
    if family in train_families:
        folder = "train"
    elif family in val_families:
        folder = "val"
    elif family in test_families:
        folder = "test"
    if folder != "":
        split.insert(-1,folder)
        new_file = "/".join(split)
        #print(f, new_file)
        #os.rename(f,new_file)
