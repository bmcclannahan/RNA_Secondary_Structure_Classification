import torch
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
from model import Model
from Siamese import SiameseNetworkDataset, Contrastive_Loss
import os

class Siamese_Model(Model):

    data_dir = "/data/Siamese"

    def __init__(self,model_func,model_name,learning_rate=0.01,lr_gamma=0.5,lr_step=50,iteration_limit=600,logging=True):
        super().__init__(model_func,model_name,learning_rate,lr_gamma,lr_step,iteration_limit,None,logging)

    def _get_criterion(self):
        return Contrastive_Loss.ContrastiveLoss()
    
    def _build_dataloaders(self):
        phases = ['train', 'val']

        print('Initializing Dataset')

        data_normalization = {
            'train': transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

        image_folders = {x: datasets.ImageFolder(os.path.join(Siamese_Model.data_dir, x), data_normalization[x]) for x in phases}

        image_datasets = {x: SiameseNetworkDataset.SiameseNetworkDataset(image_folders[x]) for x in phases}

        print('Initializing Dataloader')

        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=Model.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        self.dataloaders = dataloaders_dict