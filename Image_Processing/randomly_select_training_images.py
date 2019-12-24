from glob import glob
from random import randint

file_dir = "/data/rna_classification/train/Diff_Family/"
file_list = "/data/rna_classification/training_file_list.txt"
training_list = open(file_list,'a+')
files = glob(file_dir+"*.jpg")
num_per_family = 20

family_dict = dict()

for f in files:
    _,family1,family2,img1,img2,_ = f.split('_')
    if family1 not in family_dict.keys():
        family_dict[family1] = [img1]
    elif img1 not in family_dict[family1]:
        family_dict[family1].append(img1)
    if family2 not in family_dict.keys():
        family_dict[family2] = [img2]
    elif img2 not in family_dict[family2]:
        family_dict[family2].append(img2)
    
keys = family_dict.keys()

for key in keys:
    imgs = family_dict[key]
    at_key = 0 #so I can add 1 to i when I hit the index of the current family to prevent making same class images
    for i in len(keys)-1:
        if keys[i] == key:
            at_key = 1
        r1 = randint(0,len(imgs)-1)
        img1 = imgs[r1]
        r2 = randint(0,len(keys))
        while keys[r2] == key:
            r2 = randint(0,len(keys)-1)
        family2 = keys[r2]
        imgs2 = family_dict[key]
        r3 = randint(0,len(imgs2)-1)
        img2 = imgs2[r3]
        file_name = '_'.join([str(key),str(family2),str(img1),str(img2),"0.jpg"])
        training_list.write(file_name + '\n')
