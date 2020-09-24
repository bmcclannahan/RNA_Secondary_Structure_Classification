from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
from Siamese import Siamese_Network

import statistics
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
        
def make_weights_for_classes(images,class_weights):
    print("Class Weights:", class_weights)
    nclasses = 2
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N*class_weights[i]/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

class Model:

    data_dir = "/scratch/b523m844/RNA_Secondary_Structure_Classification/final_datasets"

    batch_size = 32
    iteration_size = {'train': 320, 'val': 12800}

    def __init__(self,model_func,model_name,learning_rate=0.01,lr_gamma=0.25,lr_step=50,iteration_limit=600,class_weights=[.67,.33],logging=True):
        self.model_func = model_func
        self.name = model_name
        self.is_inception = False
        self.learning_rate = learning_rate
        self.lr_gamma = lr_gamma
        self.lr_step = lr_step
        self.iteration_limit = iteration_limit
        self.class_weights = class_weights
        self.logging = logging

    def train_model(self):
        since = time.time()
        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        train_acc = 0.0
        best_train_acc = 0.0

        # curr_loss = 0
        # iteration_loss_count = 200
        # prev_loss = [0]*iteration_loss_count

        iteration = 1

        iteration_validation_frequency = 50
        # iteration_loss_stddev_termination_threshold = .005
        # iteration_loss_termination_threshold = .01
        iteration_count_termination_thresholid = self.iteration_limit
        

        while iteration <= iteration_count_termination_thresholid:
            if self.logging:
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
                    running_loss, running_corrects, running_total = self._train_phase(running_loss, running_corrects)
                if phase == 'val':
                    running_loss, running_corrects, running_total, class_correct, class_total = self._val_phase(running_loss,running_corrects,class_correct,class_total)

                print(running_loss,running_corrects,running_total)

                if phase == 'train':
                    iteration_loss = running_loss / Model.iteration_size['train']
                    iteration_acc = running_corrects / Model.iteration_size['train']
                    train_acc = iteration_acc
                else:
                    iteration_loss = running_loss / len(self.dataloaders[phase].dataset)
                    accuracy = list(0. for i in range(2))
                    for i in range(len(accuracy)):
                        accuracy[i] = class_correct[i]/class_total[i]
                    iteration_acc = sum(accuracy)/len(accuracy)

                # if phase == 'train':
                #     prev_loss = [curr_loss] + prev_loss[:9]
                #     curr_loss = iteration_loss

                if phase == 'val' and self.logging:
                    fp.write('{: .4f} and {: .4f}\n'.format(iteration_loss, iteration_acc))
                    #write the validation loss enough times so it can be graphed over the train loss
                    for i in range(iteration_validation_frequency):
                        fvl.write(str(iteration_loss)+"\n")
                if phase == 'train' and self.logging:
                    ft.write('{: .4f} and {: .4f}\n'.format(iteration_loss, iteration_acc))
                    ftl.write(str(iteration_loss)+"\n")

                print('{} Loss: {: .4f} Acc: {:.4f}'.format(phase, iteration_loss, iteration_acc))

                per_iteration_model = copy.deepcopy(self.model.state_dict())
                if self.logging:
                    torch.save(per_iteration_model, '/scratch/b523m844/RNA_Secondary_Structure_Classification/' + self.name + '/chekers/iter'+str(iteration)+'.pt')

                if phase == 'val' and iteration_acc > best_acc:
                    best_acc = iteration_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_train_acc = train_acc
                if phase == 'val':
                    val_acc_history.append(iteration_acc)

            print()
            
            if self.logging:
                fp.close()
                ft.close()
            iteration += 1

        time_elapsed = time.time() - since
        print('Model Name:', self.name)
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed / 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best train acc: {:4f}'.format(best_train_acc))

        self.model.load_state_dict(best_model_wts)

        if self.logging:
            torch.save(best_model_wts,'/scratch/b523m844/RNA_Secondary_Structure_Classification/' + self.name + '/checkpoints/best.pt')
        return self.model, val_acc_history

    def _train_phase(self,running_loss,running_corrects):
        for _ in range(int(Model.iteration_size['train']/Model.batch_size)):
            inputs, labels = next(iter(self.dataloaders['train']))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects.double(), Model.iteration_size['train']

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

        return running_loss, running_corrects, Model.iteration_size['val'], class_correct, class_total

    def initialize_model(self):
        final_layer_size = 2
        feature_extract = False
        use_pretrained=True
        model_ft = self.model_func(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if self.name.split('_')[0] == 'resnet':
            print("Detected resnet model")
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, final_layer_size)
        elif self.name.split('_')[0] == 'vgg':
            print("Detected vggnet model")
            num_ftrs = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_ftrs, final_layer_size)
        elif self.name.split('_')[0] == 'densenet':
            print("Detected densenet model")
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, final_layer_size)
        elif self.name.split('_')[0] == 'siamese':
            print("Detected siamese model")
            model_ft = Siamese_Network.SiameseNetwork(self.model_func)

        return model_ft


    def build_model(self):
        print("Training",self.name,"model")

        model_ft = self.initialize_model()
        print(model_ft)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()
        print("Parmas to learn:")

        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)


        optimizer_ft = optim.SGD(params_to_update, lr=self.learning_rate, momentum=0.9)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=self.lr_step, gamma=self.lr_gamma)


        criterion = self._get_criterion()

        self._build_dataloaders()

        self.model = model_ft
        self.criterion = criterion
        self.optimizer = optimizer_ft
        self.scheduler = exp_lr_scheduler
        self.device = device

    def _get_criterion(self):
        return nn.CrossEntropyLoss()

    def _build_dataloaders(self):
        phases = ['train', 'val']

        print('Initializing Dataset')

        data_normalization = {
            'train': transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(Model.data_dir, x), data_normalization[x]) for x in phases}
        print('Weighting Classes')

        weights = make_weights_for_classes(image_datasets['train'].imgs,self.class_weights)
        weights = torch.DoubleTensor(weights)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=Model.iteration_size['train'])

        print('Initializing Dataloader')

        data_loaders_train = torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=Model.batch_size, sampler=sampler_train, num_workers=2)
        data_loaders_val = torch.utils.data.DataLoader(image_datasets['val'], batch_size=Model.batch_size, num_workers=4)

        dataloaders_dict = {
            'train': data_loaders_train,
            'val':data_loaders_val
        }

        self.dataloaders = dataloaders_dict

    def _clear_dataloaders(self):
        dataloaders_dict = {
            'train': None,
            'val': None
        }
        self.dataloaders = dataloaders_dict

    def _test_model(self,model):
        self.model.load_state_dict(model)
        self.model.eval()
        
        fs = open("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/predictionhval.txt", "a")
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        
        print(time.ctime())

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
        
        print('Model Name:', self.name)
        for i in range(2):
            print('Accuracy of %5s : %3.1d %%' % (str(i), 100.0 * class_correct[i] / class_total[i])) 
        print('Total accuracy is %3.1d %%' % (100.0 * sum(class_correct) / sum(class_total)))
        print(time.ctime())

    def _test_best_model(self):
        self._test_model(torch.load("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/checkpoints/best.pt"))
    
    def _test_iteration_model(self,iteration):
        self._test_model(torch.load("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/chekers/iter" + str(iteration) +".pt"))

    def test_model(self,iterations_to_test=[199,299,399,499,599]):
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

        print("Testing Best Model")

        self._test_best_model()

        if len(iterations_to_test) > 0:
            for iteration in iterations_to_test:
                print("Testing model from iteration:",iteration)
                self._test_iteration_model(iteration)

        print("Finished testing " + self.name)