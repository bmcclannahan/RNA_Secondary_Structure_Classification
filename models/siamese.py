from model import Model
import torch.nn as nn
from torchvision import models

model_name = "siamese"

resnet50 = models.resnet50(pretrained=True)

print(model_name)

model = Model(models.resnet50,model_name)
model.build_model()
model.train_model()
model.test_model()
