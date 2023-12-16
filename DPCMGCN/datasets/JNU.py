import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from itertools import islice
from sklearn.model_selection import train_test_split
from DPCMGCN.datasets.SequenceDatasets import dataset
from DPCMGCN.datasets.sequence_aug import *
from tqdm import tqdm


signal_size = 1024

# root = 'F:\Data\JNU\data'
root = 'F:\Data\SEU\bearingset'
# root = 'F:\Data\SEU\gearset'



'''
dataname = {
    0: ["ib600_2.csv", "n600_3_2.csv", "ob600_2.csv", "tb600_2.csv"],
    1: ["ib800_2.csv", "n800_3_2.csv", "ob800_2.csv", "tb800_2.csv"],
    2: ["ib1000_2.csv", "n1000_3_2.csv", "ob1000_2.csv", "tb1000_2.csv"]
}
label = [i for i in range(0, 4)]


def get_files(root, N):  
  
    # This function is used to generate the final training set and test set.
    # root:The location of the data set
  
    data = []
    lab = []  
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label[n])
            data += data1
            lab += lab1

    # print('---------------')
    # print(len(data[0]))
    print(np.array(data).shape)
    print(np.array(lab).shape)

    return [data, lab]


def data_load(filename, label):

    # This function is mainly used to generate test data and training data.
    # filename:Data location
    # axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
   
    fl = np.loadtxt(filename)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        if len(fl[start:end]) == 1024:
            data.append(fl[start:end])
            lab.append(label)
            start += signal_size
            end += signal_size

    # print("======================")
    # aaa = np.array(data).shape
    # print(aaa)
    # print("======================")
    return data, lab

# --------------------------------------------------------------------------------------------------------------------
'''


dataname= {
    0: ["ball_20_0.csv", "comb_20_0.csv", "health_20_0.csv", "inner_20_0.csv", "outer_20_0.csv"],
    1: ["ball_30_2.csv", "comb_30_2.csv", "health_30_2.csv", "inner_30_2.csv", "outer_30_2.csv"]
}
label = [i for i in range(0, 5)]


def get_files(root, N):
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, dataname=dataname[N[k]][n], label=label[n])
            data += data1
            lab += lab1

    print(np.array(data).shape)
    print(np.array(lab).shape)

    return [data, lab]


def data_load(filename, dataname, label):
    f = open(filename, "r", encoding='gb18030', errors='ignore')
    fl = []
    if dataname == "ball_20_0.csv":
        for line in islice(f, 16, None):
            line = line.rstrip()
            word = line.split(",", 8)
            fl.append(eval(word[1]))
    else:
        for line in islice(f, 16, None):
            line = line.rstrip()
            word = line.split("\t", 8)
            fl.append(eval(word[1]))
    fl = np.array(fl)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]/10:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

# --------------------------------------------------------------------------------------------------------------------


'''
dataname= {
    0: ["Chipped_20_0.csv", "Health_20_0.csv", "Miss_20_0.csv", "Root_20_0.csv", "Surface_20_0.csv"],
    1: ["Chipped_30_2.csv", "Health_30_2.csv", "Miss_30_2.csv", "Root_30_2.csv", "Surface_30_2.csv"]
}
label = [i for i in range(0, 5)]


# generate Training Dataset and Testing Dataset
def get_files(root, N):
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, dataname=dataname[N[k]][n], label=label[n])
            data += data1
            lab += lab1

    print(np.array(data).shape)
    print(np.array(lab).shape)

    return [data, lab]


def data_load(filename, dataname, label):
    f = open(filename, "r", encoding='gb18030', errors='ignore')
    fl = []
    if dataname == "ball_20_0.csv":
        for line in islice(f, 16, None):  
            line = line.rstrip()
            word = line.split(",", 8)  
            fl.append(eval(word[1]))   
    else:
        for line in islice(f, 16, None):  
            line = line.rstrip()
            word = line.split("\t", 8)  
            fl.append(eval(word[1]))  
    fl = np.array(fl)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]/10:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

# --------------------------------------------------------------------------------------------------------------------
'''


class JNU(object):

    # num_classes = 4

    num_classes = 5

    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            list_data = get_files(self.data_dir, self.source_N)

            # list_data (2,1539)
            # list_data[0] (1539, 1024, 1)
            # list_data[1] (1539,)
            # print(np.array(list_data[0]).shape)
            # print(np.array(list_data[1]).shape)

            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            return source_train, source_val, target_train, target_val
        else:
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val

