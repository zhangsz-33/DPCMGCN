#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from torch import nn
import warnings
import torch
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.utils import dropout_adj
from models.CNN import CoordAtt


class GGL(torch.nn.Module):
    '''
    Grapg generation layer
    '''

    def __init__(self,):
        super(GGL, self).__init__()
        '''
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
        '''

        self.layer = nn.Sequential(
            nn.Linear(256, 10),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # x = self.encoder(x)
        # x = self.decoder(x)

        atrr = self.layer(x)
        values, edge_index = Gen_edge(atrr)
        return values.view(-1), edge_index


def Gen_edge(atrr):
    atrr = atrr.cpu()
    A = torch.mm(atrr, atrr.T)
    maxval, maxind = A.max(axis=1)
    A_norm = A / maxval
    k = A.shape[0]
    values, indices = A_norm.topk(k, dim=1, largest=True, sorted=False)
    edge_index = torch.tensor([[], []], dtype=torch.long)

    for i in range(indices.shape[0]):
        index_1 = torch.zeros(indices.shape[1], dtype=torch.long) + i
        index_2 = indices[i]
        sub_index = torch.stack([index_1, index_2])
        edge_index = torch.cat([edge_index, sub_index], axis=1)

    return values, edge_index


class MultiChev(torch.nn.Module):
    def __init__(self, in_channels,):
        super(MultiChev, self).__init__()
        self.scale_1 = ChebConv(in_channels, 400, K=1)
        self.scale_2 = ChebConv(in_channels, 400, K=2)
        self.scale_3 = ChebConv(in_channels, 400, K=3)

    def forward(self, x, edge_index, edge_weight):
        scale_1 = self.scale_1(x, edge_index, edge_weight)
        # (64, 400)
        # print(scale_1.detach().numpy().shape)
        scale_2 = self.scale_2(x, edge_index, edge_weight)
        scale_3 = self.scale_3(x, edge_index, edge_weight)
        return torch.cat([scale_1, scale_2, scale_3], 1)


class MultiChev_B(torch.nn.Module):
    def __init__(self, in_channels,):
        super(MultiChev_B, self).__init__()
        self.scale_1 = ChebConv(in_channels, 100, K=1)
        self.scale_2 = ChebConv(in_channels, 100, K=2)
        self.scale_3 = ChebConv(in_channels, 100, K=3)

    def forward(self, x, edge_index, edge_weight):
        scale_1 = self.scale_1(x, edge_index, edge_weight)
        scale_2 = self.scale_2(x, edge_index, edge_weight)
        scale_3 = self.scale_3(x, edge_index, edge_weight)
        return torch.cat([scale_1, scale_2, scale_3], 1)



class MGCN(nn.Module):
    '''
    This code is the implementation of MRF-GCN
    T. Li et al., "Multi-receptive Field Graph Convolutional Networks for Machine Fault Diagnosis"
    '''
    def __init__(self, pretrained=False, in_channel=256, out_channel=10):
        super(MGCN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.atrr = GGL()
        self.conv1 = MultiChev(in_channel)
        self.bn1 = BatchNorm(1200)

        self.layer = CoordAtt(1200, 1200)

        self.conv2 = MultiChev_B(400 * 3)
        self.bn2 = BatchNorm(300)

        # GAT
        self.conv3 = GATConv(in_channel, 400, heads=3, dropout=0.5)
        self.bn3 = BatchNorm(1200)
        self.conv4 = GATConv(1200, 100, heads=3, dropout=0.5)
        self.bn4 = BatchNorm(300)

        # GCN
        self.conv_GCN_1 = GCNConv(1200, 1200)
        self.bn_GCN_1 = BatchNorm(1200)
        self.conv_GCN_2 = GCNConv(1200, 300)
        self.bn_GCN_2 = BatchNorm(300)

        self.layer5 = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())

    def forward(self, x):

        edge_atrr, edge_index = self.atrr(x)
        # edge_atrr = edge_atrr.cuda()
        # edge_index = edge_index.cuda()
        edge_index, edge_atrr = dropout_adj(edge_index, edge_atrr)

        # ChebConv
        x = self.conv1(x, edge_index, edge_weight=edge_atrr)
        x = self.bn1(x)

        # x = self.conv_GCN_1(x, edge_index)
        # x = self.bn_GCN_1(x)

        x = self.conv2(x, edge_index, edge_weight=edge_atrr)
        x = self.bn2(x)
        # (64,1200)
        # print(x.detach().numpy().shape)

        # GAT
        # x, (edge_index, edge_atrr) = self.conv3(x, edge_index, return_attention_weights=True)
        # x = self.bn3(x)
        # x = self.conv4(x, edge_index)
        # x = self.bn4(x)

        # GCN
        # x = self.conv_GCN_1(x, edge_index, edge_weight=edge_atrr)
        # x = self.bn_GCN_1(x)
        # x = self.conv_GCN_2(x, edge_index, edge_weight=edge_atrr)
        # x = self.bn_GCN_2(x)


        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x
