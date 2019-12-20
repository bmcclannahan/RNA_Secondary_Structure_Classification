from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms

import statistics
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/final_datasets"

batch_size = 32
epoch_size = {'train': 320, 'val': 12800}

class ImageFolderWithPaths(datasets.ImageFolder):
     def __getitem__(self, index):
         # this is what ImageFolder normally returns 
         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
         # the image file path
         path = self.imgs[index][0]
         # make a new tuple that includes original and the pat
         tuple_with_path = (original_tuple + (path,))
         return tuple_with_path


def train_model(model, dataloaders, criterion, optimizer, schedular, device, model_name, is_inception=False):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    curr_loss = 0
    epoch_loss_count = 200
    prev_loss = [0]*epoch_loss_count

    epoch = 0

    epoch_validation_frequency = 50
    epoch_loss_stddev_termination_threshold = .005
    epoch_loss_termination_threshold = .01
    epoch_count_termination_thresholid = 1000
    

    while epoch < epoch_loss_count or (statistics.stdev([curr_loss]+prev_loss) > epoch_loss_stddev_termination_threshold 
        and curr_loss > epoch_loss_termination_threshold and epoch <= epoch_count_termination_thresholid):
        ft = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + model_name + "/train_result.txt", "a")
        fp = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + model_name + "/val_result.txt", "a")
        ftl = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + model_name + "/train_loss.txt", "a")
        fvl = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + model_name + "/val_loss.txt", "a")
        print('Epoch {}'.format(epoch))
        print(time.ctime())
        print('-' * 20)

        for phase in ['train', 'val']:
            if epoch % epoch_validation_frequency != 0 and epoch != 0 and phase == 'val':
                continue

            if phase == 'train':
                schedular.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i in range(int(epoch_size[phase]/batch_size)):
                inputs, labels = next(iter(dataloaders[phase]))
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                #class_correct = list(0. for i in range(2))
                #class_total = list(0. for i in range(2))
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / epoch_size[phase]

            epoch_acc = running_corrects.double()/epoch_size[phase]

            if phase == 'train':
                prev_loss = [curr_loss] + prev_loss[:9]
                curr_loss = epoch_loss

            if phase == 'val':
                fp.write('{: .4f} and {: .4f}\n'.format(epoch_loss, epoch_acc))
                #write the validation loss enough times so it can be graphed over the train loss
                for i in range(epoch_validation_frequency):
                    fvl.write(str(epoch_loss)+"\n")
            if phase == 'train':
                ft.write('{: .4f} and {: .4f}\n'.format(epoch_loss, epoch_acc))
                ftl.write(str(epoch_loss)+"\n")

            print('{} Loss: {: .4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            per_epoch_model = copy.deepcopy(model.state_dict())
            torch.save(per_epoch_model, '/scratch/b523m844/RNA_Secondary_Structure_Classification/' + model_name + '/chekers/iter'+str(epoch)+'.pt')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        fp.close()
        ft.close()
        epoch += 1

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s '.format(time_elapsed / 60, time_elapsed % 60))
    print('Best value Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    torch.save(best_model_wts,'/scratch/b523m844/RNA_Secondary_Structure_Classification/' + model_name + '/checkpoints/best.pt')
    return model, val_acc_history


def initialize_model(model):
    num_classes = 2
    feature_extract = False
    use_pretrained=True
    model_ft = None
    input_size = 0
    model_ft = model(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

            
def make_weights_for_classes(images):
    nclasses = 2
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    class_weight = [.5, .5]
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N*class_weight[i]/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def build_model(model):
    phases = ['train', 'val']

    model_ft, input_size = initialize_model(model)
    print(model_ft)

    print('Initializing Dataset')

    data_normalization = {
        'train': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_normalization[x]) for x in phases}
    print('Weighting Classes')

    weights = make_weights_for_classes(image_datasets['train'].imgs)
    weights = torch.DoubleTensor(weights)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=epoch_size['train'])

    print('Initializing Dataloader')

    data_loaders_train = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=batch_size, sampler=sampler_train, num_workers=4)
    data_loaders_val = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, num_workers=4)

    dataloaders_dict = {
        'train': data_loaders_train,
        'val':data_loaders_val
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Parmas to learn:")

    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)


    optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.25)


    criterion = nn.CrossEntropyLoss()

    return model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, device

def _test_model(model,dataloaders, device, model_name):
    best_model_wts = torch.load("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + model_name + "/checkpoints/best.pt") 
    model.load_state_dict(best_model_wts)
    model.eval()
    
    fs = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + model_name + "predicionhval.txt", "a")
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
        print('Accuracy of %5s : %3d %%' % (str(i), 100 * class_correct[i] / class_total[i])) 
    print('Total accuracy is %3d %%' % (100 * sum(class_correct) / sum(class_total)))

def test_model(model, model_name):
    data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/final_datasets"

    data_transforms = {
        'test': transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

    print('Initializing dataset and dataloader')

    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir,x), data_transforms[x]) for x in ['test']}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['test']} 

    print("Loading Model")

    model_ft, input_size = initialize_model(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)

    print("Testing Model")

    _test_model(model_ft, dataloaders_dict, device, model_name)