import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
from model import Model
from Siamese import SiameseNetworkDataset
import os

class Siamese_Model(Model):

    data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/final_datasets/Siamese"

    iteration_size = {'train': 1661, 'val': 120}

    def __init__(self,model_func,model_name,learning_rate=0.01,lr_gamma=0.5,lr_step=50,iteration_limit=600,logging=True):
        super().__init__(model_func,model_name,learning_rate,lr_gamma,lr_step,iteration_limit,None,logging)

    def _get_criterion(self):
        return nn.BCELoss()

    def _train_phase(self,running_loss,running_corrects):
        num_iterations = int(Siamese_Model.iteration_size['train']/Model.batch_size)
        print("Number of iterations:", num_iterations)
        for _ in range(num_iterations):
            inputs1, inputs2, labels = next(iter(self.dataloaders['train']))

            print("Number of inputs",len(inputs1),len(inputs2))

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

            running_loss += loss.item() * inputs1.size(0)
            expected = torch.reshape(labels.data,(Model.batch_size))
            running_corrects += torch.sum(preds.float() == expected)

            print("Shape of preds:",preds.shape)
            print("Shape of labels:", expected.shape)
            print("Number of labels:", len(labels))
            print("Running corrects:", running_corrects)

        return running_loss, running_corrects.int(), Siamese_Model.iteration_size['train']

    def _val_phase(self,running_loss,running_corrects,class_correct,class_total):
        for _ in range(int(Siamese_Model.iteration_size['val']/Model.batch_size)):
            inputs1, inputs2, labels = next(iter(self.dataloaders['val']))

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
            running_loss += loss.item() * inputs1.size(0)
            expected = labels.data.long()
            running_corrects += torch.sum(preds == expected)

        return running_loss, running_corrects, Siamese_Model.iteration_size['val'], None, None
    
    def _build_dataloaders(self):
        phases = ['train', 'val']

        print('Initializing Dataset')

        data_normalization = {
            'train': transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

        image_folders = {x: datasets.ImageFolder(os.path.join(Siamese_Model.data_dir, x)) for x in phases}

        image_datasets = {x: SiameseNetworkDataset.SiameseNetworkDataset(image_folders[x], data_normalization[x]) for x in phases}

        print('Initializing Dataloader')

        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=Model.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        self.dataloaders = dataloaders_dict