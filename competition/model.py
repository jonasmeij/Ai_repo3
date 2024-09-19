import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import random


class NeuralNetwork(nn.Module):

    def __init__(self):
        super.__init__()
        self.dummy_param = nn.Parameter(torch.rand(1))

        # define the layers

    def forward(self, x):
        # define the forward pass
        return x
