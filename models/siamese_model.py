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

    def _train_phase(self,running_loss,running_corrects):
        for _ in range(int(Model.iteration_size['train']/Model.batch_size)):
            inputs, labels = next(iter(self.dataloaders['train']))
            
            inputs1 = [i[0] for i in inputs]
            inputs2 = [i[1] for i in inputs]
            inputs1 = inputs1.to(self.device)
            inputs2 = inputs2.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs1,inputs2)
                
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects

    def _val_phase(self,running_loss,running_corrects,class_correct,class_total):
        for inputs, labels in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)

                
                loss = self.criterion(outputs, labels)       
                _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            for i in range(len(labels)):
                class_correct[labels[i]] += int(labels[i] == preds[i])
                class_total[labels[i]] += 1

        return running_loss, running_corrects, class_correct, class_total
    
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