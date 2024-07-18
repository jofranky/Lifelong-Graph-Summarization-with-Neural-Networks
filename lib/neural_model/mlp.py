"""
mlp.py: mlp model based on torch.nn.Linear layers
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear  


class MLP(torch.nn.Module):
    def __init__(self, num_features,num_classes, hidden_channels = 32, dropout = 0.5 ):
        """
        if hidden_channels == 0, then a 1-layer/hop model is built,
        else a 2 layer model with 1 hidden_channel is built
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        if hidden_channels == 0:
            self.lin1 = Linear(num_features, num_classes)
        else:
            self.lin1 = Linear(num_features, hidden_channels)
            self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, data):  
        if self.hidden_channels == 0:
            x = self.lin1(data.x)
        else:
            x = self.lin1(data.x)
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout, training=self.training)
            x = self.lin2(x)
        return F.log_softmax(x, dim=1)