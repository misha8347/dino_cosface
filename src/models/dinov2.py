import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.models.cosface import CosFace

num_classes = 30
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dinov2_vitb14)
        self.classifier = CosFace(num_classes=num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x