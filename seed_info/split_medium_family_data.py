import os, glob, shutil

def copy_files(files,family_dict):
    for f in files:
        split = f.split('/')
        family_name = split[-1][:7]
        family = int(family_name[2:])
        if family in family_dict.keys():
            split.insert(-1,dest_folder)
            new_file = "/".join(split)
            #print(f, new_file)
            shutil.copyfile(f,new_file)

seed_data = "/users/b523m844/RNA_Secondary_Structure_Classification/seed_info/Medium_families.txt"
source_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/Full_Image_Set/"
dest_folder = "Medium_Families"

with open(seed_data) as f:
    medium_families = f.readlines()
medium_families = [x.rstrip() for x in medium_families]

family_dict = dict()

for i in range(len(medium_families)):
    split = medium_families[i].split(",")
    family_dict[int(split[0][2:])] = split[1]

files = glob.glob(source_dir+"*.jpg")

print("Number of files:",len(files))
print("Number of medium families:",len(family_dict.keys()))
keys = list(family_dict.keys())
print("First 10 keys:",keys[:10])

copy_files(files,family_dict)






