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

    def __init__(self,model_func,model_name,learning_rate=0.01,lr_gamma=0.25,lr_step=50,iteration_limit=300,start_weights=[.2,.8],end_weights[.8,.2]):
        super().__init__(model_func,model_name,learning_rate,lr_gamma,lr_step,iteration_limit,start_weights)
        self.start_weights = start_weights
        self.end_weights = end_weights
    
    def execute_training_steps():
        self.build_model()
        self.train_model()
        self.class_weights = end_weights
        self.build_model()
        self.model = torch.load("/scratch/b523m844/RNA_Secondary_Structure_Classification/" + self.name + "/chekers/iter" + str(self.iteration_limit) +".pt")
        self.train_model()
        self.test_model(iterations_to_test=[])