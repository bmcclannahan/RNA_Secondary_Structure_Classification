import os, glob

rna_list = '/data/train.out'
data_dir = '/data/Full_Image_Set/'

images = glob.glob(data_dir+"*.jpg")

rna = []

rna = open(rna_list)
lines = rna.readlines()

k = 0

while k < len(lines):
    line = lines[k]
    if line.split(' ') == 'family':
        rna.append(lines[k+2].split(' ')[-1])    
    k += 4

print(rna)