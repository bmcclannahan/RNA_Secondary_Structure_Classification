from torchvision import models

from model import test_model

model_name = "resnet_101"

test_model(models.resnet101,model_name)