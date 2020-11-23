import torch
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image
import datetime

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transforms=None,weight=0.5,mode='train'):
        self.imageFolderDataset = imageFolderDataset
        self.transforms = transforms
        self.weight = weight
        self.seed = datetime.datetime.utcnow().second
        self.image_count = 0
        print("Building dataset for", mode)
        # if mode == 'train':
        #     self.mode = False
        # else:
        #     self.mode = True
        #     self._intialize_dataset()
        self.mode = False

    def _intialize_dataset(self):
        image_dict = {}
        for image,label in self.imageFolderDataset.imgs:
            if label not in image_dict.keys():
                image_dict[label] = [image]
            else:
                image_dict[label].append(image)
        
        keys = list(image_dict.keys())
        self.image_list = []

        for current_key in range(len(keys)):
            key = keys[current_key]
            same_images = []
            different_images = []
            family_size = len(image_dict[key])
            for i in range(family_size):
                for j in range(i,family_size):
                    same_images.append([image_dict[key][i],image_dict[key][j],False])
            for i in range(current_key+1,len(keys)):
                new_key = keys[i]
                for j in range(family_size):
                    new_family_size = len(image_dict[new_key])
                    for k in range(new_family_size):
                        different_images.append([image_dict[key][j],image_dict[new_key][k],True])
            self.image_list.extend(different_images)
            self.image_list.extend(same_images)
        print("Dataset Size:", len(self.image_list))

    def get_dataset_size(self):
        return len(self.image_list)

    def load_images_directly(self,index,batch_size):
        img0_list = [0]*batch_size
        img1_list = [0]*batch_size
        label_list = [0]*batch_size

        for i in range(batch_size):
        img0_path, img1_path, label = self.image_list[index]
        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        
        if self.transforms is not None:#I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
        
            img0_list[i] = img0
            img1_list[i] = img1
            label_list[i] = torch.from_numpy(np.array([label],dtype=np.float32))

        return img0_list, img1_list, label_list

    def __getitem__(self,index):
        if self.mode:
            img0_path, img1_path, label = self.image_list[index]

        else:
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            #we need to make sure approx 50% of images are in the same class
            should_get_same_class = random.random()
            if should_get_same_class < self.weight:
                while True:
                    #keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                    if img0_tuple[1]==img1_tuple[1]:
                        break
            else:
                while True:
                    #keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                    if not img0_tuple[1]==img1_tuple[1]:
                        break

            img0_path = img0_tuple[0]
            img1_path = img1_tuple[0]
            label = img0_tuple[1] != img1_tuple[1]

        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')

        
        if self.transforms is not None:#I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
        
        return img0, img1, torch.from_numpy(np.array([label],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)