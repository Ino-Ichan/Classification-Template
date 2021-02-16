import torch
import torch.nn as nn
from torchvision import transforms, models

import timm

class Net(nn.Module):
    def __init__(self, name="resnest101e", n_classes=1):
        super(Net, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=11)

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x
