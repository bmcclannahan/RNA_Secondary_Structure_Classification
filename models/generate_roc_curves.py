from model import Model
from siamese_model import Siamese_Model
import torch.nn as nn
from torchvision import models

weights = [.5,.5]

model_name = "resnet_50"
print(model_name, weights)
model = Model(models.resnet50,model_name,class_weights=weights)
model.build_model()
model.test_model(iterations_to_test=[])

model_name = "resnet_101"
print(model_name, weights)
model = Model(models.resnet101,model_name,class_weights=weights)
model.build_model()
model.test_model(iterations_to_test=[])

model_name = "vgg_19"
print(model_name, weights)
model = Model(models.vgg19,model_name,class_weights=weights)
model.build_model()
model.test_model(iterations_to_test=[])

model_name = "siamese_r50"
print(model_name, weights)
model = Siamese_Model(models.resnet50,model_name,logging=True)
model.build_model()
model.test_model(iterations_to_test=[])

model_name = "siamese_r101"
print(model_name, weights)
model = Siamese_Model(models.resnet101,model_name,logging=True)
model.build_model()
model.test_model(iterations_to_test=[])

model_name = "siamese_v19"
print(model_name)
model = Siamese_Model(models.vgg19,model_name,logging=True,model_type='vggnet')
model.build_model()
model.test_model(iterations_to_test=[])