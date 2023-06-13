import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
from classes import Model,Patience,select_para, experiment
import json

# IMDB-BINARY
para = {0: {'Batch': 4, 'learning_rate': 0.01, 'hidden_size': 64},
        1: {'Batch': 4, 'learning_rate': 0.001, 'hidden_size': 64},
        2: {'Batch': 4, 'learning_rate': 0.01, 'hidden_size': 32},
        3: {'Batch': 4, 'learning_rate': 0.001, 'hidden_size': 32},
        4: {'Batch': 8, 'learning_rate': 0.01, 'hidden_size': 64},
        5: {'Batch': 8, 'learning_rate': 0.001, 'hidden_size': 64},
        6: {'Batch': 8, 'learning_rate': 0.01, 'hidden_size': 32},
        7: {'Batch': 8, 'learning_rate': 0.001, 'hidden_size': 32}}


para_groups = 8

if __name__ == "__main__":
    start_time = time.time()  
    m0 = 4
    m = m0 ** 2
    R = 3
    K = 10
    nums_eopch = 500
    early_stop_use_loss = False
    patience_nums = 50
    classnum = 2
    # output_loss = "./log/REDDIT-BINARY/loss.txt"
    # output_acc = "./log/REDDIT-BINARY/acc.txt"
    datasetsplit = "datasets/IMDB-BINARY/IMDB-BINARY_1_splits"
    datax = pd.read_csv('datasets/IMDB-BINARY/IMDB-BINARY_m4.csv',
        sep=',', header=None)
    label = pd.read_csv('datasets/IMDB-BINARY/IMDB-BINARY/IMDB-BINARY_graph_labels.csv',
        header=None)
    datax = (datax - datax.mean()) / (datax.std())

    res = []
    # split
    with open("{}.json".format(datasetsplit), "r", encoding="utf-8") as f:
        content = json.load(f)

    for idx, item in enumerate(content):
        test_idx = item['test']
        tmp = item['model_selection'][0]
        train_valid = []
        for tmp_ in tmp:
            train_valid.append(tmp[tmp_])
        train_idx, valid_idx = train_valid[0], train_valid[1]

        train_df, train_label = datax.iloc[train_idx], label.iloc[train_idx]
        test_df, test_label = datax.iloc[test_idx], label.iloc[test_idx]
        valid_df, valid_label = datax.iloc[valid_idx], label.iloc[valid_idx]

        # tensor
        train_data = torch.tensor(train_df.values).to(torch.float32)
        train_label = torch.tensor(train_label.values)
        test_data = torch.tensor(test_df.values).to(torch.float32)
        test_label = torch.tensor(test_label.values)
        valid_data = torch.tensor(valid_df.values).to(torch.float32)
        valid_label = torch.tensor(valid_label.values)

        # tensordataset
        data_arrays_train = (train_data, train_label)
        dataset_train = data.TensorDataset(*data_arrays_train)
        data_arrays_test = (test_data, test_label)
        dataset_test = data.TensorDataset(*data_arrays_test)
        data_arrays_valid = (valid_data, valid_label)
        dataset_valid = data.TensorDataset(*data_arrays_valid)

        SelectPara = select_para()
        for i in range(para_groups):
            batch = para[i]['Batch']
            lr = para[i]['learning_rate']
            hidden = para[i]['hidden_size']

            inner_model = experiment(classnum,m,batch,lr,hidden,nums_eopch,dataset_train,dataset_valid)
            val_loss,val_accu,train_loss,acc_list = inner_model.train_valid_earlystop(early_stop_use_loss,patience_nums)
            SelectPara.update(val_accu, batch, lr, hidden,train_loss,acc_list)

        batch, lr, hidden,train_loss,acc_list = SelectPara.get_para()
        print("best para: ",batch,lr,hidden)

        perf_r = 0
        for r in range(R):
            outer_model = experiment(classnum,m, batch, lr, hidden, nums_eopch, dataset_train, dataset_valid)
            val_loss, val_accu, train_loss, val_accu_list = outer_model.train_valid_earlystop(early_stop_use_loss,
                                                                                              patience_nums)
            print("R = ", r, 'val accu ', val_accu)
            cur = outer_model.get_accu(dataset_test)
            print("R = ", r, 'accu ', cur)
            perf_r = perf_r + cur
        perf_r = perf_r / R
        print("perf_r", perf_r)
        res.append(perf_r)

    end_time = time.time()
    run_time = end_time - start_time
    print("time：", run_time)
    print("acc: ", res)
