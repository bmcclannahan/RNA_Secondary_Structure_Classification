from model import Model
import torch.nn as nn
from torchvision import models

model_name = "densenet_161"

weights = [.5,.5]

print(model_name, weights)

model = Model(models.densenet161,model_name,class_weights=weights)
model.build_model()
model.train_model()
model.test_model()
