"""
utils.py: Wrapper functions for the modules torch and pandas.
"""

import torch
import torch.nn.functional as F
import torch_geometric

import timeit

import pandas as pd




def Ncontrast(x_dis, adj_label, tau = 1):
    """
    Compute the Ncontrast loss
    Functions for GraphMLP see https://github.com/yanghu819/Graph-MLP

    
    Args:
        x_dis (torch.Tensor): Intermediate result layer
        adj_label (torch.Tensor): Adjacency matrix 
        tau (int): Temperature ( default is 1 )
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss
    
def get_A_r(data, r):
    """
    Convert adjaceny list to matrix and precalculate adjacency matrix based on k-hop

    
    Args:
        data (torch_geometric.data.Data): Mini batch
        r (int): K-hop
    """
    adj = torch.sparse.FloatTensor(
            data.edge_index, 
            torch.ones(data.edge_index.shape[1]), 
            [data.x.shape[0],data.x.shape[0]])
    adj_m = adj.to_dense()
    adj_m[adj_m!=0]=1.0
    adj_label = adj_m.fill_diagonal_(1.0)
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label

