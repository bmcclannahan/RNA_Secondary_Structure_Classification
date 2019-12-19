from model import train_model, build_model, set_parameter_requires_grad
import torch.nn as nn
from torchvision import models

model_name = "resnet"

def initialize_model():
    num_classes = 2
    feature_extract = False
    use_pretrained=True
    model_ft = None
    input_size = 0
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, device = build_model(initialize_model)

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                             optimizer_ft, exp_lr_scheduler, device=device, is_inception=False)
