from model import Model
import torch.nn as nn
from torchvision import models

model_name = "resnet_101"

model = Model(models.resnet101,model_name)
model.build_model()
model.train_model()
model.test_model()
