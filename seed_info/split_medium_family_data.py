import os, glob, shutil

def copy_files(files,family_list,folder):
    for f in files:
        split = f.split('/')
        family_name = split[-1][:7]
        family = int(family_name[2:])
        if family in family_list:
            split.insert(-1,folder)
            new_file = "/".join(split)
            #print(f, new_file)
            shutil.copyfile(f,new_file)

train_data = "/users/b523m844/RNA_Secondary_Structure_Classification/seed_info/Train.txt"
test_data = "/users/b523m844/RNA_Secondary_Structure_Classification/seed_info/Test.txt"
val_data = "/users/b523m844/RNA_Secondary_Structure_Classification/seed_info/Val.txt"
source_dir = "/home/b523m844/data/Full_Image_Set/"
dest_folder = "Medium_Small_Families/"

with open(train_data) as f:
    train_families = f.readlines()
train_families = [int(x.rstrip()[2:]) for x in train_families]
with open(test_data) as f:
    test_families = f.readlines()
test_families = [int(x.rstrip()[2:]) for x in test_families]
with open(val_data) as f:
    val_families = f.readlines()
val_families = [int(x.rstrip()[2:]) for x in val_families]

files = glob.glob(source_dir+"*.jpg")

print("Number of files:",len(files))
print("Number of medium small families:",len(train_families)+len(test_families)+len(val_families))

copy_files(files,train_families,dest_folder+"train")
copy_files(files,test_families,dest_folder+"test")
copy_files(files,val_families,dest_folder+"val")






