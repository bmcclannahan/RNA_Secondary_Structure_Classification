import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
from model import Model

import statistics
import matplotlib.pyplot as plt
import time
import os
import copy

class Multi_Training_Phase_Model(Model):

    def __init__(self,model_func,model_name,learning_rate=0.001,lr_gamma=0.5,lr_step=50,iteration_limit=600,iteration_swap_threshold=350,start_weights=[.2,.8],end_weights=[.8,.2]):
        super().__init__(model_func,model_name,learning_rate,lr_gamma,lr_step,iteration_limit,start_weights)
        self.start_weights = start_weights
        self.end_weights = end_weights
        self.iteration_swap_threshold = iteration_swap_threshold
    
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

        while iteration <= self.iteration_limit:
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
                    train_acc = iteration_acc
                else:
                    iteration_loss = running_loss / len(self.dataloaders[phase].dataset)
                    accuracy = list(0. for i in range(2))
                    for i in range(len(accuracy)):
                        accuracy[i] = class_correct[i]/class_total[i]
                    iteration_acc = sum(accuracy)/len(accuracy)


                if phase == 'val':
                    fp.write('{: .4f} and {: .4f}\n'.format(iteration_loss, iteration_acc))
                    #write the validation loss enough times so it can be graphed over the train loss
                    for i in range(iteration_validation_frequency):
                        fvl.write(str(iteration_loss)+"\n")
                if phase == 'train':
                    ft.write('{: .4f} and {: .4f}\n'.format(iteration_loss, iteration_acc))
                    ftl.write(str(iteration_loss)+"\n")

                print('{} Loss: {: .4f} Acc: {:.4f}'.format(phase, iteration_loss, iteration_acc))
                for param in self.optimizer.param_groups:
                    print('Learning Rate:', param['lr'])
                

                per_iteration_model = copy.deepcopy(self.model.state_dict())
                torch.save(per_iteration_model, '/scratch/b523m844/RNA_Secondary_Structure_Classification/' + self.name + '/chekers/iter'+str(iteration)+'.pt')

                if phase == 'val' and iteration_acc > best_acc:
                    best_acc = iteration_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_train_acc = train_acc
                if phase == 'val':
                    val_acc_history.append(iteration_acc)

                if iteration == self.iteration_swap_threshold and phase == 'val':
                    self.class_weights = self.end_weights
                    self._build_dataloaders()
                    # for param in self.optimizer.param_groups:
                    #     param['lr'] = .001

            print()
            fp.close()
            ft.close()
            iteration += 1

        time_elapsed = time.time() - since
        print('Model Name:', self.name)
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed / 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best train acc: {:4f}'.format(best_train_acc))

        self._clear_dataloaders()

        self.model.load_state_dict(best_model_wts)

        torch.save(best_model_wts,'/scratch/b523m844/RNA_Secondary_Structure_Classification/' + self.name + '/checkpoints/best.pt')
        return self.model, val_acc_history