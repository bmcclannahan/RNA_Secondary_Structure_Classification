import torch.nn as nn
from torchvision import models
from siamese_model import Siamese_Model

model_name = "siamese_v19"

print(model_name)

model = Siamese_Model(models.vgg19,model_name,logging=True,model_type='vggnet')
model.build_model()
model.train_model()
print("Validation image count:",model.dataloaders['val'].get_dataset_size())
model.test_model(iterations_to_test=[])
print("Model func: VGGNet 19")
