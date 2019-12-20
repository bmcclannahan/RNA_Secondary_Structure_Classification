from model import train_model, build_model
import torch.nn as nn
from torchvision import models

model_name = "resnet_101"

model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, device = build_model(models.resnet101)

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                             optimizer_ft, exp_lr_scheduler, device, model_name, is_inception=False)