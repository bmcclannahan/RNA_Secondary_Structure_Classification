import torch
from torch.utils.data import Dataset
import random
import numpy as np

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset):
        self.imageFolderDataset = imageFolderDataset
        
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

        img0 = img0_tuple[0]
        img1 = img1_tuple[0]
        
        return img0, img1, torch.from_numpy(np.array([should_get_same_class],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)