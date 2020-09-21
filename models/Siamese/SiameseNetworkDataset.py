import torch
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transforms=None):
        self.imageFolderDataset = imageFolderDataset
        self.transforms = transforms
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0_path = img0_tuple[0]
        img1_path = img1_tuple[0]

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        
        if self.transform is not None:#I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        print(type(img0),type(img1))

        if self.transforms is not None:
            for transform in self.transforms:
                img0 = transform(img0)
                img1 = transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([should_get_same_class],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)