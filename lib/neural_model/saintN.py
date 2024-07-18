"""
saint.py: based on GraphSAINT model which is based on torch_geometric.nn.GCNConv layers 
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SAINTN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=32, dropout=0.5,h_layers=0):
        """
        if hidden_channels == 0, then a 1-layer/hop model is built,
        else a 2 layer model with hidden_channels is built
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.h_layers = h_layers
        if hidden_channels == 0 or self.h_layers == 0:
            self.conv1 = GCNConv(num_node_features, num_classes)
        elif self.h_layers == 1:
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(2 * hidden_channels, num_classes)
        else:
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(3 * hidden_channels, num_classes)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr

    def forward(self, data, edge_weight=None):
        if self.hidden_channels == 0 or self.h_layers == 0:
            x = self.conv1( data.x, data.edge_index, edge_weight )
        elif self.h_layers == 1:
            x1 = F.relu(self.conv1(data.x, data.edge_index, edge_weight))
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.relu(self.conv2(x1, data.edge_index, edge_weight))
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x = torch.cat([x1, x2], dim=-1)
            x = self.lin(x)
        else:
            x1 = F.relu(self.conv1(data.x, data.edge_index, edge_weight))
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.relu(self.conv2(x1, data.edge_index, edge_weight))
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x3 = F.relu(self.conv3(x2, data.edge_index, edge_weight))
            x3 = F.dropout(x3, p=self.dropout, training=self.training)
            x = torch.cat([x1, x2,x3], dim=-1)
            x = self.lin(x)
        return F.log_softmax( x, dim=-1 )  
