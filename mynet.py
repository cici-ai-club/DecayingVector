import torch.nn as nn
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
class network(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(network, self).__init__()
        self.hidden_size = hidden_size
        self.output_hidden_size = output_size
        self.fmap = nn.Sequential(nn.Linear(input_size, hidden_size),nn.Linear(hidden_size,  hidden_size//2), nn.ReLU())
        self.linear = nn.Linear(hidden_size//2, self.output_hidden_size)
    def forward(
            self, 
            inputs=None,
            labels = None):
        output1 = self.fmap(inputs)
        output = self.linear(output1)
        if labels is not None: # when we have labels
            loss_fct2 = CrossEntropyLoss()
            sloss = loss_fct2(output, labels)
            loss = sloss
            return output, loss
        else: # when we do not have labels
            return output
