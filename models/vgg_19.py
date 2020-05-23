from model import Model
import torch.nn as nn
from torchvision import models

model_name = "vgg_19"

weights = [.8,.2]

print(model_name, weights)

model = Model(models.vgg19,model_name,learning_rate=.005,iteration_limit=700,class_weights=weights)
model.build_model()
model.train_model()
model.test_model(iterations_to_test=[299,399,499,599,699])
