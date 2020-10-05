import torch
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transforms=None,weight=0.5):
        self.imageFolderDataset = imageFolderDataset
        self.transforms = transforms
        self.weight = weight
        self.seed = 1
        self.image_count = 0
        
    def __getitem__(self,index):
        if self.image_count > 1500:
            self.seed += 1    
            random.seed(self.seed)    
        self.image_count += 1
        
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.random()
        if should_get_same_class < self.weight:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        elif:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if not img0_tuple[1]==img1_tuple[1]:
                    break

        img0_path = img0_tuple[0]
        img1_path = img1_tuple[0]

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')
        
        if self.transforms is not None:#I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
        
        return img0, img1, torch.from_numpy(np.array([img0_tuple[1] == img1_tuple[1]],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)