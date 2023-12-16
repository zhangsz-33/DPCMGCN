#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim
from DPCMGCN.utils.lr_scheduler import *
import DPCMGCN.models as models
import DPCMGCN.datasets as datasets
from DPCMGCN.utils.save import Save_Tool
from DPCMGCN.loss.DAN import DAN
from sklearn import svm
import numpy as np

import matplotlib.pyplot as plt
# from sklearn import datasets
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix




def plot_tsne(features, labels, domain_label, title):
    '''
    features:(N*m)
    label:(N)
    '''

    tsne = TSNE(n_components=2, init='pca', random_state=0)

    class_num = len(np.unique(labels))
    latent = features
    tsne_features = tsne.fit_transform(features)
    # print('tsne_features shape:', tsne_features.shape)
    # plt.scatter(tsne_features[:, 0], tsne_features[:, 1])
    # plt.show()

    df = pd.DataFrame()
    df["y"] = labels
    df['d'] = domain_label
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    # plt.scatter(x="comp-1", y="comp-2", )
    # plt.show()

    plt.figure()
    # plt.scatter(x="comp-1", y="comp-2", s=70)
    ax = sns.scatterplot(x="comp-1", y="comp-2", s=300, hue=df.y.tolist(), style=df.d.tolist(),
                         palette=sns.color_palette("hls", class_num),
                         data=df).set(title="{}".format(title))

    l = plt.legend()
    # l.get_texts()[0].set_text('inner')  # You can also change the legend title
    # l.get_texts()[1].set_text('health')
    # l.get_texts()[2].set_text('outer')
    # l.get_texts()[3].set_text('ball')
    # 东南大学
    l.get_texts()[0].set_text('ball')  # You can also change the legend title
    l.get_texts()[1].set_text('comb')
    l.get_texts()[2].set_text('health')
    l.get_texts()[3].set_text('inner')
    l.get_texts()[4].set_text('outer')

    # palette = sns.color_palette("hls", class_num)

    # cbar = plt.colorbar()
    # ticks = [0, 1, 2, 3]
    # ticklabels = ['IB', 'Normal', 'OB', 'TB']
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(ticklabels)
    plt.show()


def plot_tsne1(features, y, d, title=None):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_features = tsne.fit_transform(features)
    X = tsne_features
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()



def plot_confusion_matrix(True_label, T_predict1, batch_idx):
    f, ax = plt.subplots()
    C = confusion_matrix(True_label, T_predict1)
    C1 = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
    C1 = C1.round(2)
    # xtick = ['IB', 'NO', 'OB', 'TB']
    # ytick = ['IB', 'NO', 'OB', 'TB']
    xtick = ['ball', 'comb', 'health', 'inner', 'outer']
    ytick = ['ball', 'comb', 'health', 'inner', 'outer']

    h = sns.heatmap(C1, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=xtick, yticklabels=ytick)
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=10)
    ax.set_xlabel('predict(1->0)')
    ax.set_ylabel('true(1->0)')
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)

    # ax.set_ylim(len(C1)-0.5, -0.5)
    plt.savefig('C:/10{}.png'.format(batch_idx))
    plt.show()




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):

        args = self.args

        if torch.cuda.is_available():
            '''
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
            '''
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        Dataset = getattr(datasets, args.data_name)
        print(Dataset)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
            # print(args.transfer_task)
            args.transfer_task = eval("".join(args.transfer_task))

        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}


        self.model = getattr(models, args.model_name)(args.pretrained)
        # print(self.model)
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)

        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders['source_train'])*(args.max_epoch-args.middle_epoch)
            self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                        hidden_size=args.hidden_size, max_iter=self.max_iter)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.domain_adversarial:
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        if args.domain_adversarial:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
        else:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]

        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        elif args.lr_scheduler == 'reduce':
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        elif args.lr_scheduler == 'transferLearning':
            param_lr = []
            for param_group in self.optimizer.param_groups:
                param_lr.append(param_group["lr"])
            self.lr_scheduler = transferLearning(self.optimizer, param_lr, args.max_epoch)
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            # print("======================")
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model_all.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model_all.load_state_dict(torch.load(args.resume, map_location=args.device))


        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)


        self.adversarial_loss = nn.BCELoss()

        self.structure_loss = DAN

        self.criterion = nn.CrossEntropyLoss()

        # self.criterion = LSR()
        # self.early_stopping = EarlyStopping(patience=15, verbose=True)

    def text_save(self, filename, data):
        file = open(filename, 'a')
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')
            s = s.replace("'", '').replace(',', '') + '\n'
            file.write(s)
        file.close()
        print("SUCESS")

    def train(self):
        acclog = []
        losslog = []

        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        save_list = Save_Tool(max_num=args.max_model_num)
        iter_num = 0
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)


            # Update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
                # self.lr_scheduler.step(epoch_loss)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
                # logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
                # logging.info('current lr: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
            else:
                logging.info('current lr: {}'.format(args.lr))
                

            iter_target = iter(self.dataloaders['target_train'])
            # print(iter_target)
            len_target_loader = len(self.dataloaders['target_train'])
            for phase in ['source_train', 'source_val', 'target_val']:
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    predict_label = []
                    true_label = []
                    # plot_tsne(inputs.view(inputs.size(0), -1), labels, 'Input data')
                    # print(batch_idx)
                    if phase != 'source_train' or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        source_labels = labels
                        target_inputs, target_labels = iter_target.next()
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        all_labels = torch.cat((source_labels, target_labels), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        # print('===============')
                        # train = np.array(inputs).shape
                        # print(train)
                        # print('================')

                        # print(np.array(inputs))
                        # inputs (64,1,1024)
                        # if phase == 'source_train':
                        #     features = self.model(inputs)
                        #     plot_tsne(features.detach().numpy(), labels, 'Output data')
                        # else:
                        features = self.model(inputs)
                        # plot_tsne(features.detach().numpy())
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        # plot_tsne(features.detach().numpy(), labels, 'Output data')

                        # clf = svm.SVC(C=0.9, kernel='linear')  # linear kernel
                        # clf.fit(outputs.detach().numpy(), labels)

                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            # plot_tsne(logits.detach().numpy(), labels, 'Output data')
                            loss = self.criterion(logits, labels)

                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            # print(logits)
                            # print(labels)
                            # plot_tsne(logits.detach().numpy(), labels, 'Output data')
                            classifier_loss = self.criterion(logits, labels)

                        # -----------------------------------------------------
                        if phase == 'source_train' and epoch >= args.middle_epoch:

                            # Calculate the domain adversarial
                            if args.domain_adversarial:
                                domain_label_source = torch.ones(labels.size(0)).float()
                                domain_label_target = torch.zeros(inputs.size(0)-labels.size(0)).float()
                                adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                adversarial_out = self.AdversarialNet(features)
                                # plot_tsne(features.detach().numpy(), all_labels, adversarial_label, 'Output data')
                                # plot_tsne(inputs.view(inputs.size(0), -1), all_labels, adversarial_label, 'Input data')
                                adversarial_loss = self.adversarial_loss(adversarial_out, adversarial_label)
                                structure_loss = self.structure_loss(features.narrow(0, 0, labels.size(0)),
                                                                features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)))

                                if args.trade_off_adversarial == 'Cons':
                                    lam_adversarial = args.lam_adversarial
                                elif args.trade_off_adversarial == 'Step':
                                    lam_adversarial = 2 / (1 + math.exp(-10 * ((epoch-args.middle_epoch) /
                                                                            (args.max_epoch-args.middle_epoch)))) - 1
                                else:
                                    raise Exception("loss not implement")

                                loss = classifier_loss + lam_adversarial * adversarial_loss + lam_adversarial * structure_loss
                            else:
                                loss = classifier_loss

                        pred = logits.argmax(dim=1)
                        predict_label.extend(pred.numpy().tolist())
                        true_label.extend(labels.numpy().tolist())
                        # if epoch == 299:
                        #     plot_confusion_matrix(labels, pred, batch_idx)
                        correct = torch.eq(pred, labels).float().sum().item()

                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # plot_confusion_matrix(predict_label, true_label, epoch)



                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                # acclog.append(epoch_acc)
                # losslog.append(epoch_loss)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                # if self.lr_scheduler is not None:
                #     # self.lr_scheduler.step(epoch)
                #     self.lr_scheduler.step(epoch_loss)
                #     logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
                # else:
                #     logging.info('current lr: {}'.format(args.lr))

                if phase == 'target_val':
                    acclog.append(epoch_acc)
                    losslog.append(epoch_loss)
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if (epoch_acc > best_acc or epoch > args.max_epoch-2) and (epoch > args.middle_epoch-1):
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

        acc_file_name = 'C:/Users/SEU-CNN-acc.txt'
        loss_file_name = 'C:/Users/SEU-CNN-loss.txt'
        self.text_save(acc_file_name, acclog)
        self.text_save(loss_file_name, losslog)

            #         if epoch > 100:
            #             self.early_stopping(epoch_loss / epoch_length, self.model_all)
            # if self.early_stopping.early_stop:
            #     print("Early stopping")
            #     break















