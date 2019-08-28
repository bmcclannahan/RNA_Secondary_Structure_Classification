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
from pylab import *
import time
import os
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


class ImageFolderWithPaths(datasets.ImageFolder):
     def __getitem__(self, index):
         # this is what ImageFolder normally returns 
         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
         # the image file path
         path = self.imgs[index][0]
         # make a new tuple that includes original and the pat
         tuple_with_path = (original_tuple + (path,))
         return tuple_with_path




input_size = 224

batch_size = 32
num_classes = 2
feature_extract = False
data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/Big_Training_Set"

data_transforms = {
     'test': transforms.Compose([
      transforms.RandomResizedCrop(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

print('Initializing dataset and dataloader')

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir,x), data_transforms[x]) for x in ['test']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['test']} 


def test_model(model,dataloaders):
    best_model_wts = torch.load("/scratch/b523m844/RNA_Secondary_Structure_Classification/resnet/chekers/epoch99.pt") 
    model.load_state_dict(best_model_wts)
    model.eval()
    
    
    fs = open("predicionhval.txt", "a")
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    for inputs, labels, path in dataloaders['test']:
        
    
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _,preds = torch.max(outputs,1)
        c = (preds == labels).squeeze()
        path_list = list(path)
        l = 0
        for item in path_list:
            pt = str(item)
            labs = labels[l]
            lb = labs.item()
            k = preds[l]
            pn = k.item()
            sn = str(pn)
            ls = str(lb)
            op = outputs[l]
            opn = op.cpu().detach().numpy()
            a1 = str(opn[0])
            a2 = str(opn[1])

            
            l = l + 1
            fs.write(pt + " " + sn + " " + ls + " " + a1 + " " + a2 + "\n")
            
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        
    fs.close()
    
    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
        str(i), 100 * class_correct[i] / class_total[i])) 
  

def initialize_model(model_name, num_classes, feature_extract, use_pretrained = False):
 
     model_ft = None
     input_size = 0

     if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

     return model_ft, input_size

model_name = "resnet50"


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False 

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

test_model(model_ft, dataloaders_dict)



"""
def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))


"""
   
        
        
        
   
    


  
      

