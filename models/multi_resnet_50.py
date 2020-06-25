from multi_training_phase_model import Multi_Training_Phase_Model
import torch.nn as nn
from torchvision import models

model_name = "multi_resnet_50"

weights = [.8,.5]#

print(model_name, weights)

model = Multi_Training_Phase_Model(models.resnet50,model_name)
model.execute_training_steps()