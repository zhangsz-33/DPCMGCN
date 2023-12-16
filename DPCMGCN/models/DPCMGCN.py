#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch
from DPCMGCN.models.MGCN import MGCN
from DPCMGCN.models.CNN import CNN, decoder, encoder, resnet18, BiLSTM
from DPCMGCN.models.CNN import CoordAtt


class DPCMGCN(nn.Module):
    def __init__(self, pretrained=False):
        super(DPCMGCN, self).__init__()
        self.model_cnn = CNN(pretrained)
        # self.model_ae1 = encoder(pretrained)
        # self.model_ae2 = decoder(pretrained)
        # self.model_resnet = resnet18(pretrained)
        # self.model_Bilstm = BiLSTM()
        self.model_GCN = MGCN(pretrained)

        self.__in_features = 256*1

    def forward(self, x):
        x1 = self.model_cnn(x)  # CNN
        # x1 = self.model_ae1(x)  # AE
        # x2 = self.model_ae2(x1)  # AE
        # x2 = self.model_resnet(x)  # resnet
        # x2 = self.model_Bilstm(x)
        # print(x1.detach().numpy().shape)
        # x1 = CoordAtt(256, 256)
        x2 = self.model_GCN(x1)
        return x2

    def output_num(self):
        return self.__in_features
