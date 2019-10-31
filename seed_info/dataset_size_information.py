from glob import glob

folder = "/home/b523m844/data/Full_Image_Set/Medium_Small_Families/train/"

file_dict = dict()

files = glob(folder+"*.jpg")
for f in files:
    family = int(f.split('/')[-1][2:7])
    if family not in file_dict.keys():
        file_dict[family] = 0
    file_dict[family] += 1

same = 0
diff = 0
families = file_dict.items()

for family,count in families:
    for f,c in families:
        if family == f:
            same += count*(c-1)
        else:
            diff += count*c


print("Count for folder:", folder)
print("Total:", diff+same)
print("Same:", same)
print("Diff:", diff)