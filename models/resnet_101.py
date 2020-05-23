from model import Model
import torch.nn as nn
from torchvision import models

model_name = "resnet_101"

weights = [.5,.5]

print(model_name, weights)

model = Model(models.resnet101,model_name,class_weights=weights)
model.build_model()
model.train_model()
model.test_model()
