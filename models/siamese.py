from Siamese import Siamese_Network
import torch.nn as nn
from torchvision import models
from siamese_model import Siamese_Model

model_name = "siamese"

resnet50 = models.resnet50(pretrained=True)

print(model_name)

model = Siamese_Model(models.resnet50,model_name,logging=True)
#model.build_model()
#model.train_model()
model.test_model(iterations_to_test=[14,24,34,44])
