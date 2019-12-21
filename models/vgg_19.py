from model import Model
import torch.nn as nn
from torchvision import models

model_name = "vgg_19"

model = Model(models.vgg19,model_name)
model.build_model()
model.train_model()
model.test_model()
