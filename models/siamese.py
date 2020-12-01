import torch.nn as nn
from torchvision import models
from siamese_model import Siamese_Model

model_name = "siamese"

print(model_name)

#model = Siamese_Model(models.resnet50,model_name,logging=True)
model = Siamese_Model(models.resnet101,model_name,logging=True)
#model = Siamese_Model(models.vgg19,model_name,logging=True)
model.build_model()
model.train_model()
print("Validation image count:",model.dataloaders['val'].get_dataset_size())
model.test_model(iterations_to_test=[])
#print("Model func: ResNet 50")
print("Model func: ResNet 101")
#print("Model func: VGGNet 19")
