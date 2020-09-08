import os, glob
import shutil

rna_list = '/data/train.out'
data_dir = '/data/Full_Image_Set/'
dest_dir = '/data/Siamese/train/'

images = glob.glob(data_dir+"*.jpg")

rna_fmailies = []

rna = open(rna_list)
lines = rna.readlines()

k = 0

while k < len(lines):
    line = lines[k]
    if line.split(' ')[0] == 'family':
        family = int(lines[k+2].split(' ')[-1].rstrip())
        rna_fmailies.append(family)
        print(family)
    k += 4

print(rna_fmailies)

print('Number of RNA:',len(images))

for image in images:
    print(image)
    print(image.split('/')[-1][2:].split('_')[0])
    family = int(image.split('/')[-1][2:].split('_')[0])
    if family in rna_fmailies:
        new_file = image.split('/')[-1]
        #print(new_file)
        shutil.copy(image,dest_dir + new_file)