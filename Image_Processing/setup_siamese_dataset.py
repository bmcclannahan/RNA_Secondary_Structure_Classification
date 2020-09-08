import os, glob

rna_list = '/data/train.out'
data_dir = '/data/Full_Image_Set/'

images = glob.glob(data_dir+"*.jpg")

rna_fmailies = []

rna = open(rna_list)
lines = rna.readlines()

k = 0

while k < len(lines):
    line = lines[k]
    if line.split(' ') == 'family':
        family = lines[k+2].split(' ')[-1]
        rna_fmailies.append(family)
        print(family)
    k += 4

print(rna_fmailies)