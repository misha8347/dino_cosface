import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

feature_dim = 768
num_classes = 30
s = 30.0
m = 0.8

class CosFace(nn.Module):
    def __init__(self, num_features=feature_dim, num_classes=num_classes, s=s, m=m):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, features, labels=None):
        # Normalize features and weight matrix
        features = F.normalize(features)
        W = F.normalize(self.W)
        
        # Compute cosine similarity
        logits = F.linear(features, W)
        
        if labels is not None:
            one_hot_labels = F.one_hot(labels, num_classes=self.W.size(0)).float()
            logits_m = logits - one_hot_labels * self.m
            logits = torch.where(one_hot_labels.byte(), logits_m, logits)

        return logits * self.s
