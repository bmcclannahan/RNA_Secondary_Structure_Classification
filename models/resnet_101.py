from model import train_model, build_model
import torch.nn as nn
from torchvision import models

model_name = "resnet_101"

model = Model(models.resnet101,model_name)
model.build_model()
model.train_model()
model.test_model()