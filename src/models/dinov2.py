import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.models.cosface import CosFace

dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes, num_features, s, m):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dinov2_vitb14)
        self.classifier = CosFace(num_features=num_features, num_classes=num_classes,
                                  s=s, m=m)

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x