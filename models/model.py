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

class ImageFolderWithPaths(datasets.ImageFolder):
     def __getitem__(self, index):
         # this is what ImageFolder normally returns 
         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
         # the image file path
         path = self.imgs[index][0]
         # make a new tuple that includes original and the pat
         tuple_with_path = (original_tuple + (path,))
         return tuple_with_path


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

class Model:

    data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/final_datasets"

    batch_size = 32
    iteration_size = {'train': 320, 'val': 12800}

    def __init__(self,model_func,model_name):
        self.model_func = model_func
        self.name = model_name
        self.is_inception = False

    def train_model(self):
        since = time.time()
        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # curr_loss = 0
        # iteration_loss_count = 200
        # prev_loss = [0]*iteration_loss_count

        iteration = 1

        iteration_validation_frequency = 50
        # iteration_loss_stddev_termination_threshold = .005
        # iteration_loss_termination_threshold = .01
        iteration_count_termination_thresholid = 400
        

        while iteration <= iteration_count_termination_thresholid:
            ft = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/train_result.txt", "a")
            fp = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/val_result.txt", "a")
            ftl = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/train_loss.txt", "a")
            fvl = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/val_loss.txt", "a")
            print('Iteration {}'.format(iteration))
            print(time.ctime())
            print('-' * 20)

            for phase in ['train', 'val']:
                if iteration % iteration_validation_frequency != 0 and iteration != 0 and phase == 'val':
                    continue

                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0
                
                class_correct = list(0. for i in range(2))
                class_total = list(0. for i in range(2))

                if phase == 'train':
                    for i in range(int(Model.iteration_size[phase]/Model.batch_size)):
                        inputs, labels = next(iter(self.dataloaders[phase]))
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        self.optimizer.zero_grad()
                        # class_correct = list(0. for i in range(2))
                        # class_total = list(0. for i in range(2))
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)

                            loss = self.criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)

                            loss.backward()
                            self.optimizer.step()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        self.optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                        
                            loss = self.criterion(outputs, labels)       
                            _, preds = torch.max(outputs, 1)
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        for i in range(len(labels)):
                            class_correct[labels[i]] += int(labels[i] == preds[i])
                            class_total[labels[i]] += 1

                if phase == 'train':
                    iteration_loss = running_loss / Model.iteration_size['train']
                    iteration_acc = running_corrects.double() / Model.iteration_size['train'] 
                else:
                    iteration_loss = running_loss / len(self.dataloaders[phase].dataset)
                    accuracy = list(0. for i in range(2))
                    for i in range(len(accuracy)):
                        accuracy[i] = class_correct[i]/class_total[i]
                    iteration_acc = sum(accuracy)/len(accuracy)

                # if phase == 'train':
                #     prev_loss = [curr_loss] + prev_loss[:9]
                #     curr_loss = iteration_loss

                if phase == 'val':
                    fp.write('{: .4f} and {: .4f}\n'.format(iteration_loss, iteration_acc))
                    #write the validation loss enough times so it can be graphed over the train loss
                    for i in range(iteration_validation_frequency):
                        fvl.write(str(iteration_loss)+"\n")
                if phase == 'train':
                    ft.write('{: .4f} and {: .4f}\n'.format(iteration_loss, iteration_acc))
                    ftl.write(str(iteration_loss)+"\n")

                print('{} Loss: {: .4f} Acc: {:.4f}'.format(phase, iteration_loss, iteration_acc))

                per_iteration_model = copy.deepcopy(self.model.state_dict())
                torch.save(per_iteration_model, '/scratch/b523m844/RNA_Secondary_Structure_Classification/' + self.name + '/chekers/iter'+str(iteration)+'.pt')

                if phase == 'val' and iteration_acc > best_acc:
                    best_acc = iteration_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(iteration_acc)

            print()
            fp.close()
            ft.close()
            iteration += 1

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s '.format(time_elapsed / 60, time_elapsed % 60))
        print('Best value Acc: {:4f}'.format(best_acc))

        self.model.load_state_dict(best_model_wts)

        torch.save(best_model_wts,'/scratch/b523m844/RNA_Secondary_Structure_Classification/' + self.name + '/checkpoints/best.pt')
        return self.model, val_acc_history

    def initialize_model(self):
        num_classes = 2
        feature_extract = False
        use_pretrained=True
        model_ft = None
        input_size = 0
        model_ft = self.model_func(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if self.name.split('_')[0] == 'resnet':
            print("Detected resnet model")
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        elif self.name.split('_')[0] == 'vgg':
            print("Detected vggnet model")
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        return model_ft, input_size


    def build_model(self):
        phases = ['train', 'val']

        model_ft, input_size = self.initialize_model()
        print(model_ft)

        print('Initializing Dataset')

        data_normalization = {
            'train': transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(Model.data_dir, x), data_normalization[x]) for x in phases}
        print('Weighting Classes')

        weights = make_weights_for_classes(image_datasets['train'].imgs)
        weights = torch.DoubleTensor(weights)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=Model.iteration_size['train'])

        print('Initializing Dataloader')

        data_loaders_train = torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=Model.batch_size, sampler=sampler_train, num_workers=4)
        data_loaders_val = torch.utils.data.DataLoader(image_datasets['val'], batch_size=Model.batch_size, num_workers=4)

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

        self.model = model_ft
        self.dataloaders = dataloaders_dict
        self.criterion = criterion
        self.optimizer = optimizer_ft
        self.scheduler = exp_lr_scheduler
        self.device = device

    def _test_model(self):
        best_model_wts = torch.load("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/checkpoints/best.pt") 
        self.model.load_state_dict(best_model_wts)
        self.model.eval()
        
        fs = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/predictionhval.txt", "a")
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))

        for inputs, labels, path in self.dataloaders['test']:
            
        
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
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

    def test_model(self):
        data_transforms = {
            'test': transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            }

        print('Initializing dataset and dataloader')

        image_datasets = {x: ImageFolderWithPaths(os.path.join(Model.data_dir,x), data_transforms[x]) for x in ['test']}

        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=Model.batch_size, shuffle=True, num_workers=4) for x in ['test']} 

        print("Loading Model")

        model_ft, input_size = self.initialize_model()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_ft = model_ft.to(device)

        print("Testing Model")

        self._test_model()

        print("Finished testing " + self.name)