from model import train_model, build_model
import torch.nn as nn
from torchvision import models

model_name = "resnet_50"

model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, device = build_model(models.resnet50,model_name)

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                             optimizer_ft, exp_lr_scheduler, device, model_name, is_inception=False)
