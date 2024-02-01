#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch
from models.MGCN import MGCN
from models.CNN import CNN, decoder, encoder, resnet18, BiLSTM, TransformerEncoder, DSRN, rsnet18
from models.CNN import CoordAtt


class DPCMGCN_features(nn.Module):
    def __init__(self, pretrained=False):
        super(DPCMGCN_features, self).__init__()
        self.model_cnn = CNN(pretrained)
        # self.model_transformer = TransformerEncoder()
        # self.model_ae1 = encoder(pretrained)
        # self.model_ae2 = decoder(pretrained)
        # self.model_resnet = resnet18(pretrained)
        # self.model_Bilstm = BiLSTM()
        self.model_GCN = MGCN(pretrained)
        # self.model_DSRN = DSRN()
        # self.model_rsnet18 = rsnet18()
        self.__in_features = 256*1

    def forward(self, x):
        x1 = self.model_cnn(x)  # CNN
        # x1 = self.model_DSRN(x) # DSRN
        # x1 = self.model_rsnet18(x)  # rsnet18
        # x1 = self.model_transformer(x)  # Transformer
        # x1 = self.model_ae1(x)  # AE
        # x2 = self.model_ae2(x1)  # AE
        # x2 = self.model_resnet(x)  # resnet
        # x2 = self.model_Bilstm(x)
        # print(x1.detach().numpy().shape)
        # x1 = CoordAtt(256, 256)
        x1 = self.model_GCN(x1)
        return x1

    def output_num(self):
        return self.__in_features
