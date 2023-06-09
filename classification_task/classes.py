import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time


class Model(nn.Module):
    """
    A single-layer MLP with ReLU activitions
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        y_pred = self.net(x)
        return y_pred


class Patience:
    """
    Early stop
    """

    def __init__(self, patience=20, use_loss=True):
        self.local_val_optimum = float("inf") if use_loss else -float("inf")
        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss, self.train_acc = None, None
        self.val_loss, self.val_acc = None, None
        self.test_loss, self.test_acc = None, None

    def stop(self, val_loss=None, val_acc=None):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc >= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_acc
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience


class select_para:
    """
    Select the best parameters
    """
    def __init__(self):
        self.batch = 0
        self.lr = 0
        self.hidden = 0
        self.local_accur_optim = -1
        self.train_loss_list = []
        self.acc_list = []

    def update(self, val_accu, cur_batch, cur_lr, cur_hidden, cur_train_loss, cur_acc_list):
        if (val_accu > self.local_accur_optim):
            self.batch = cur_batch
            self.lr = cur_lr
            self.hidden = cur_hidden
            self.local_accur_optim = val_accu
            self.train_loss_list = cur_train_loss
            self.acc_list = cur_acc_list

    def get_para(self):
        return self.batch, self.lr, self.hidden, self.train_loss_list, self.acc_list


class experiment:
    def __init__(self, classnum, m, batch, lr, hidden, epoch, train_dataset, valid_dataset):
        self.classnum = classnum
        self.m = m
        self.batch = batch
        self.lr = lr
        self.hidden = hidden
        self.nums_epoch = epoch
        self.train_iter = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True)
        self.valid_iter = data.DataLoader(valid_dataset, batch_size=self.batch, shuffle=False)

        self.net = Model(self.m, self.hidden, self.classnum)
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr)

    def train_valid_earlystop(self, early_stop_use_loss, patience_nums):
        """
        Training with early stop
        :param early_stop_use_loss: whether to use loss for early stop
        :param patience_nums: training stops if patience_nums epochs have passed without improvement on the validation set
        :return: accuracy on the validation set, train_loss, the histories of accuracy on the validation set
        """
        EarlyStop = Patience(patience=patience_nums, use_loss=early_stop_use_loss)
        train_loss = []
        val_accu_list = []
        for epoch in range(self.nums_epoch):
            # 训练、测试
            acc_loss_train = 0
            for batch, (train_x, train_y) in enumerate(self.train_iter):
                l = self.loss(self.net(train_x), train_y.reshape(-1))
                train_loss.append(l.item())
                self.optim.zero_grad()
                l.backward()
                self.optim.step()

            val_loss = 0

            correct = 0
            for batch, (test_x, test_y) in enumerate(self.valid_iter):
                # for X, y in train_iter:
                l = self.loss(self.net(test_x), test_y.reshape(-1))
                val_loss = val_loss + l.item()
                for (XX, yy) in zip(test_x, test_y):
                    y_pred = self.net(XX)
                    y_pred = y_pred.argmax(dim=0)

                    correct += int(y_pred == yy.data)

            val_accu = (correct / len(self.valid_iter.dataset))

            val_accu_list.append(val_accu)
            # val_loss, val_accu = get_valid_acc_loss(test_iter)
            if (EarlyStop.stop(val_loss=val_loss, val_acc=val_accu)):
                break

        return val_loss, val_accu, train_loss, val_accu_list

    def get_accu(self, test_dataset):
        """
        Get the accuracy
        :param test_dataset: test set
        :return: accuracy on the test set
        """
        test_iter = data.DataLoader(test_dataset, batch_size=self.batch, shuffle=False)

        correct = 0

        for _, (test_x, test_y) in enumerate(test_iter):
            for (XX, yy) in zip(test_x, test_y):
                y_pred = self.net(XX)
                y_pred = y_pred.argmax(dim=0)
                correct += int(y_pred == yy.data)

        acc = (correct / len(test_iter.dataset))
        return acc
