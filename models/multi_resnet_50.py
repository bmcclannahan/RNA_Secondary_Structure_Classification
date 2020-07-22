from multi_training_phase_model import Multi_Training_Phase_Model
import torch.nn as nn
from torchvision import models

model_name = "testing"

weights = [.8,.2]

print(model_name, weights)

model = Multi_Training_Phase_Model(models.resnet50,model_name)
model.build_model()
model.train_model()
model.test_model(iterations_to_test=[299,399,499,599])