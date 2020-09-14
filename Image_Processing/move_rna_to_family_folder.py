import os, glob
import shutil

data_dir = '/data/Siamese/val/'

images = glob.glob(data_dir+"*.jpg")

for image in images:
    family = int(image.split('/')[-1][2:].split('_')[0])
    if not os.path.exists(data_dir+str(family)):
        os.makedirs(data_dir+str(family))
        #print('new folder:', family)
    new_file = image.split('/')[-1]
    print(new_file)
    shutil.move(image,data_dir + str(family) + '/' + new_file)