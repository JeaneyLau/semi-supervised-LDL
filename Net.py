import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from torch.nn import init

class ResNet50_semi_distribution(nn.Module):
    def __init__(self):
        super(ResNet50_semi_distribution, self).__init__()
        
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc_cls = nn.Linear(2048, 4)
        self.fc_cou = nn.Linear(2048, 65)
        
        self.fc_cls.apply(weights_init)
        self.fc_cou.apply(weights_init)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(x.size(0),2048)
        cls = self.fc_cls(features)
        cou = self.fc_cou(features)
        
        cls = F.softmax(cls,dim=1) 
        cou = F.softmax(cou,dim=1) 
        
        cou2cls = torch.stack((torch.sum(cou[:, :5], 1), torch.sum(cou[:, 5:20], 1), torch.sum(cou[:, 20:50], 1),
                               torch.sum(cou[:, 50:], 1)), 1)
        
        return features,cls,cou,cou2cls

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        if m.bias is not None:
            init.constant(m.bias, 0)