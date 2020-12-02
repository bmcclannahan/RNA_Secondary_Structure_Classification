import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
from model import Model
from Siamese import SiameseNetworkDataset as SND
#from Siamese import TestSiameseNetworkDatset as TSND
import os
import time

class Siamese_Model(Model):

    data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/final_datasets/Siamese/Full_Data"
    # data_dir = "/data/Siamese/"

    iteration_size = {'train': 3200, 'val': 16000, 'test': 64000}

    def __init__(self,model_func,model_name,learning_rate=0.001,lr_gamma=0.1,lr_step=300,iteration_limit=500,validation_frequency=25,logging=True,starting_weight=.5,model_type='resnet'):
        super().__init__(model_func,model_name,learning_rate,lr_gamma,lr_step,iteration_limit,None,validation_frequency,logging,model_type)
        self.starting_weights = {'train':starting_weight, 'val':0.5, 'test':0.5}

    def _get_criterion(self):
        return nn.BCELoss()

    def _train_phase(self,running_loss,running_corrects):
        num_iterations = int(Siamese_Model.iteration_size['train']/Model.batch_size)
        #print("Number of iterations:", num_iterations)
        for _ in range(num_iterations):
            inputs1, inputs2, labels = next(iter(self.dataloaders['train']))

            #print("Number of inputs",len(inputs1),len(inputs2))

            inputs1 = inputs1.to(self.device)
            inputs2 = inputs2.to(self.device)
            labels = labels.to(self.device)


            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs1,inputs2)
                
                loss = self.criterion(outputs, labels)
                preds, _ = torch.max(outputs, 1)
                #print("preds:", preds)
                preds = torch.round(preds)

                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * inputs1.size(0)
            expected = torch.reshape(labels.data,(Model.batch_size,))
            running_corrects += torch.sum(preds == expected)

            #print("Expected:",expected)
            #print("Outputs:", outputs)
            #print("preds:", preds)
            #print("equivalency of exp and preds:", preds == expected)
            #print("Sum of above:", torch.sum(preds == expected))
            #print("Shape of preds:",preds.shape)
            #print("Shape of labels:", expected.shape)
            #print("Number of labels:", len(labels))
            #print("Running corrects:", running_corrects)

        return running_loss, running_corrects.int().item(), Siamese_Model.iteration_size['train']

    def _val_phase(self,running_loss,running_corrects,class_correct,class_total):
        range_length = self.dataloaders['val'].get_dataset_size()//self.batch_size

        for index in range(range_length):
            inputs1, inputs2, labels = self.dataloaders['val'].load_images_directly(index*self.batch_size,self.batch_size)
            
            inputs1 = inputs1.to(self.device)
            inputs2 = inputs2.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs1,inputs2)
                
                loss = self.criterion(outputs, labels)
                preds, _ = torch.max(outputs, 1)
                preds = torch.round(preds)

            running_loss += loss.item() * inputs1.size(0)
            expected = torch.reshape(labels.data,(Model.batch_size,))
            correct = torch.sum(preds == expected)
            running_corrects += correct
            #print("Correct:", correct, "Total:", len(preds))
            #print("Running Correct:", running_corrects)

        return running_loss, running_corrects.int().item(), self.dataloaders['val'].get_dataset_size(), None, None

    def _build_dataloaders(self):
        phases = ['train', 'val']

        print('Initializing Dataset')

        data_normalization = {
            'train': transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

        image_datasets = {x: self._get_rna_dataset(x,data_normalization[x]) for x in phases}
        #image_datasets = {x: self._get_test_dataset(x,data_normalization[x]) for x in phases}

        print('Initializing Dataloader')

        dataloaders_dict = dict()
        dataloaders_dict['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=Model.batch_size, shuffle=True, num_workers=4)
        dataloaders_dict['val'] = image_datasets['val']

        self.dataloaders = dataloaders_dict

    def _get_rna_dataset(self,phase,data_normalization):
        image_folder = datasets.ImageFolder(os.path.join(Siamese_Model.data_dir,phase))
        return SND.SiameseNetworkDataset(image_folder,data_normalization,self.starting_weights[phase],phase)
    
    #For testing model works with cifar 10
    def _get_test_dataset(self,phase,data_normalization):
        image_folder = datasets.CIFAR10('/data/test_datasets')
        return SND.SiameseNetworkDataset(image_folder,data_normalization,self.starting_weights[phase],phase)

    def _test_model(self,model):
        self.model.load_state_dict(model)
        self.model.eval()
        
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        
        print(time.ctime())
        print("Testing model on",self.dataloaders['test'].get_dataset_size(),"images")

        range_length = self.dataloaders['test'].get_dataset_size()//self.batch_size

        for index in range(range_length):
            inputs1, inputs2, labels = self.dataloaders['test'].load_images_directly(index*self.batch_size,self.batch_size)
            
            inputs1 = inputs1.to(self.device)
            inputs2 = inputs2.to(self.device)
            labels = labels.to(self.device)
            labels,_ = torch.max(labels,1)
            # print('labels:',labels)
            # print(len(labels))
            outputs = self.model(inputs1,inputs2)
            preds,_ = torch.max(outputs,1)
            preds = torch.round(preds)
            # print('preds:',preds)
            # print(len(preds))
            c = (preds == labels).squeeze()
            # print("c:",c)
            # print(len(c))
            # print(c[0].item())
                
            for i in range(len(c)):
                label = int(labels[i].item())
                class_correct[label] += c[i].item()
                class_total[label] += 1
            print("Tested ", index+1,"/",range_length,"batches", end='\r',flush=True)
        

        # print(class_correct)
        # print(class_total)
        print('Model Name:', self.name)
        for i in range(2):
            print('Accuracy of %5s : %3.2f %%' % (str(i), 100.0 * class_correct[i] / class_total[i])) 
        print('Total accuracy is %3.2f %%' % (100.0 * sum(class_correct) / sum(class_total)))
        print(time.ctime())
        
    def _build_test_dataloader(self):
        data_normalization = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_dataset = self._get_rna_dataset('test',data_normalization)
        return {x: image_dataset for x in ['test']} 