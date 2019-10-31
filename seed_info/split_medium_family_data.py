import os, glob, shutil

def copy_files(files,family_dict,folder):
    for f in files:
        split = f.split('/')
        family_name = split[-1][:7]
        family = int(family_name[2:])
        if family in family_dict.keys():
            split.insert(-1,folder)
            new_file = "/".join(split)
            #print(f, new_file)
            shutil.copyfile(f,new_file)

train_data = "~/RNA_Secondary_Structure_Classification/seed_info/Train.txt"
test_data = "~/RNA_Secondary_Structure_Classification/seed_info/Test.txt"
val_data = "~/RNA_Secondary_Structure_Classification/seed_info/Val.txt"
source_dir = "/home/b523m844/data/Full_Image_Set/"
dest_folder = "Medium_Small_Families/"

with open(train_data) as f:
    train_families = f.readlines()
train_families = [x.rstrip() for x in train_families]
with open(test_data) as f:
    test_families = f.readlines()
test_families = [x.rstrip() for x in test_families]
with open(val_data) as f:
    val_families = f.readlines()
val_families = [x.rstrip() for x in val_families]

train_dict = dict()
test_dict = dict()
val_dict = dict()

for i in range(len(train_data)):
    split = train_data[i].split(",")
    train_dict[int(split[0][2:])] = split[1]
for i in range(len(test_data)):
    split = test_data[i].split(",")
    test_dict[int(split[0][2:])] = split[1]
for i in range(len(val_data)):
    split = val_data[i].split(",")
    val_dict[int(split[0][2:])] = split[1]

files = glob.glob(source_dir+"*.jpg")

print("Number of files:",len(files))
print("Number of medium small families:",len(train_families)+len(test_families)+len(val_families))

copy_files(files,train_dict,dest_folder+"train")
copy_files(files,test_dict,dest_folder+"test")
copy_files(files,val_dict,dest_folder+"val")






