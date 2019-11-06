from glob import glob
import random
import os

folder = "/home/b523m844/data/Full_Image_Set/Medium_Small_Families/train/"
threshold = 30

file_dict = dict()

files = glob(folder+"*.jpg")
for f in files:
    family = int(f.split('/')[-1][2:7])
    if family not in file_dict.keys():
        file_dict[family] = 0
    file_dict[family] += 1

families = file_dict.items()

files_to_remove = []

for family,count in families:
    if count > threshold:
        files = list(range(count))
        random.shuffle(files)
        for f in files[threshold:]:
            files_to_remove.append((family,f))

def get_file_name_from_family_and_number(t):
    family, num = t
    filename = 'RF'+str(family).zfill(5)+'_'+str(num)+'.jpg'
    return folder+filename

flist = list(map(get_file_name_from_family_and_number,files_to_remove))
# print(flist[0])
for f in flist:
    os.remove(f)