from torchvision import models

from model import test_model

model_name = "vgg_19"

test_model(models.vgg19,model_name)