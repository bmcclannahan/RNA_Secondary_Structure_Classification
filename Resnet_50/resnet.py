from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms


import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/Full_Image_Set/Medium_Families/"

model_name = "resnet"

num_classes = 2

batch_size = 32

feature_extract = False


def train_model(model, dataloaders, criterion, optimizer, schedular, is_inception=False):
   since = time.time()
   val_acc_history = []
 
   best_model_wts = copy.deepcopy(model.state_dict())
   best_acc = 0.0
   curr_loss = 0.0
   prev_loss = [1.0]*10
   epoch = 0

   while prev_loss[-1] - curr_loss <= 0.001 and prev_loss[0] - curr_loss <= 0.001
      ft = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/resnet/train_result.txt", "a") 
      fp = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/resnet/test_result.txt","a")
      print('Epoch {}'.format(epoch)
      print('-' * 10)
      
      for phase in ['train', 'val']:
         if phase == 'train':
            schedular.step()
            model.train()
         else:
            model.eval()
            
         running_loss = 0.0
         running_corrects = 0

         for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
               
            optimizer.zero_grad()
            class_correct = list(0. for i in range(2))
            class_total = list(0. for i in range(2))
            with torch.set_grad_enabled(phase == 'train'):
               outputs = model(inputs)
               
               loss = criterion(outputs, labels)       
               _, preds = torch.max(outputs, 1)
                  
               if phase == 'train':
                  loss.backward()
                  optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
         epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
         epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)
         
         if phase == 'val':
            fp.write('{: .4f} and {: .4f}\n'.format(epoch_loss, epoch_acc))
         if phase == 'train':
            ft.write('{: .4f} and {: .4f}\n'.format(epoch_loss, epoch_acc))

         print('{} Loss: {: .4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

         per_epoch_model = copy.deepcopy(model.state_dict())
         torch.save(per_epoch_model, '/scratch/b523m844/RNA_Secondary_Structure_Classification/resnet/chekers/epoch'+str(epoch)+'.pt')
            
         if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
         if phase == 'val':
            val_acc_history.append(epoch_acc)

         if phase == 'train':
            for i in [9,8,7,6,5,4,3,2,1,]:
               prev_loss[i] =  prev_loss[i-1]
            prev_loss[0] = curr_loss
            curr_loss = epoch_loss

         print()
         fp.close()
         ft.close() 
    
   time_elapsed = time.time() - since
   """print('Training complete in {:.of}m {:.0f}s '.format(time_elapsed / 60, time_elapsed % 60))"""
   print('Best value Acc: {:4f}'.format(best_acc))
 
   model.load_state_dict(best_model_wts)
    
   torch.save(best_model_wts, '/scratch/b523m844/RNA_Secondary_Structure_Classification/resnet/checkpoints/weight.pt')
   return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
   if feature_extracting:
      for param in model.parameters():
         param.requires_grad = False 



def initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):
 
     model_ft = None
     input_size = 0

    
     if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

     return model_ft, input_size

def make_weights_for_classes(images):
   nclasses = 2
   count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight  


model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print(model_ft)

# data_transforms = {
#      'train': transforms.Compose([
#       transforms.RandomResizedCrop(input_size),
#       transforms.RandomHorizontalFlip(),
#       transforms.ToTensor(),
#       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#       'val': transforms.Compose([
#       transforms.Resize(input_size),
#       transforms.CenterCrop(input_size),
#       transforms.ToTensor(),
#       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#       ])
#     }

print('Initializing dataset and dataloader')

data_normalization = {
     'train': transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
      'val': transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

      ])
    }

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x)) for data_normalization[x] in ['train', 'val']}
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train', 'val']}

weights_dict = {x: make_weights_for_classes(image_datasets[x]) for x in ['train', 'val']}
weights_dict = {x: torch.DoubleTensor(weights_dict[x]) for x in ['train','val']}
sampler_dict = {x: torch.utils.data.sampler.WeightedRandomSampler(weights_dict[x],2) for x in ['train','val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, sampler=sampler_dict[x], num_workers=4) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Parmas to learn:")

if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
     for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
           print("\t",name)


optimizer_ft = optim.SGD(params_to_update, lr = 0.01, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=70, gamma=0.1)


criterion = nn.CrossEntropyLoss()

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, is_inception = False )














