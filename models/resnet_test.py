from torchvision import models

from model import test_model

model_name = "resnet_50"

test_model(models.resnet50,model_name)