import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
from classes import Model,Patience,select_para, experiment


# REDDIT-BINARY
para = {0: {'Batch': 16, 'learning_rate': 0.01, 'hidden_size': 64},
        1: {'Batch': 16, 'learning_rate': 0.001, 'hidden_size': 64},
        2: {'Batch': 16, 'learning_rate': 0.01, 'hidden_size': 32},
        3: {'Batch': 16, 'learning_rate': 0.001, 'hidden_size': 32},
        4: {'Batch': 8, 'learning_rate': 0.01, 'hidden_size': 64},
        5: {'Batch': 8, 'learning_rate': 0.001, 'hidden_size': 64},
        6: {'Batch': 8, 'learning_rate': 0.01, 'hidden_size': 32},
        7: {'Batch': 8, 'learning_rate': 0.001, 'hidden_size': 32}}


para_groups = 8

if __name__ == "__main__":
    """
    Use a 10-fold CV for model assessment and use hold-out for parameters selection
    """
    m0 = 4
    m = m0 ** 2  # the size of the input features
    R = 3
    K = 10  # K-FOLD
    nums_eopch = 500
    holdout_train_ratio = 0.9
    outer_earlystop = 0.9
    early_stop_use_loss = False
    patience_nums = 50
    # output_loss = "./log/REDDIT-BINARY/loss.txt"
    # output_acc = "./log/REDDIT-BINARY/acc.txt"

    datax = pd.read_csv('./datasets/REDDIT-BINARY/convexrelaxation/convex_data_m4.csv',
        sep=',', header=None)
    label = pd.read_csv('./datasets/REDDIT-BINARY/REDDIT-BINARY_graph_labels.csv',
        header=None)

    res = []
    for outer_iter in range(10):
        # K-FOLD
        folds = KFold(n_splits=K, shuffle=True, random_state=42)
        perf = 0
        for trn_idx, test_idx in folds.split(datax, label):
            train_df, train_label = datax.iloc[trn_idx], label.iloc[trn_idx]
            test_df, test_label = datax.iloc[test_idx], label.iloc[test_idx]

            train_outer_data = torch.tensor(train_df.values).to(torch.float32)
            train_outer_label = torch.tensor(train_label.values.reshape(-1, 1)).to(torch.float32)
            test_outer_data = torch.tensor(test_df.values).to(torch.float32)
            test_outer_label = torch.tensor(test_label.values.reshape(-1, 1)).to(torch.float32)

            data_arrays_outer_train = (train_outer_data, train_outer_label)
            dataset_outer_train = data.TensorDataset(*data_arrays_outer_train)
            data_arrays_outer_test = (test_outer_data, test_outer_label)
            dataset_outer_test = data.TensorDataset(*data_arrays_outer_test)

            # select:  hold-out
            train_inner_dataset, valid_inner_dataset = data.random_split(dataset_outer_train,[int(len(dataset_outer_train) * holdout_train_ratio),len(dataset_outer_train) - int(len(dataset_outer_train) * holdout_train_ratio)]
                                                                         , generator=torch.Generator().manual_seed(42))

            SelectPara = select_para()
            for i in range(para_groups):
                batch = para[i]['Batch']
                lr = para[i]['learning_rate']
                hidden = para[i]['hidden_size']

                inner_model = experiment(m,batch,lr,hidden,nums_eopch,train_inner_dataset,valid_inner_dataset)
                val_accu,train_loss,acc_list = inner_model.train_valid_earlystop(early_stop_use_loss,patience_nums)
                SelectPara.update(val_accu, batch, lr, hidden,train_loss,acc_list)

            batch, lr, hidden,train_loss,acc_list = SelectPara.get_para()
            # with open(output_loss, 'a') as f_train_los:
            #     f_train_los.write("Select para\n\n")
            #     f_train_los.write(str(train_loss))
            #     f_train_los.write("\n\n\n")
            # with open(output_acc, 'a') as f_train_los:
            #     f_train_los.write("Select para\n\n")
            #     f_train_los.write(str(acc_list))
            #     f_train_los.write("\n\n\n")

            perf_r = 0
            train_R_dataset, valid_R_dataset = data.random_split(dataset_outer_train, [int(len(dataset_outer_train) * outer_earlystop),
                len(dataset_outer_train) - int(len(dataset_outer_train) * outer_earlystop)]
                 , generator=torch.Generator().manual_seed(42))

            for r in range(R):
                outer_model = experiment(m,batch,lr,hidden,nums_eopch,train_R_dataset,valid_R_dataset)
                val_accu,train_loss,val_accu_list = outer_model.train_valid_earlystop(early_stop_use_loss,patience_nums)
                cur = outer_model.get_accu(dataset_outer_test)
                perf_r = perf_r + cur

                # with open(output_loss, 'a') as f_train_los:
                #     f_train_los.write("R\n\n")
                #     f_train_los.write(str(train_loss))
                #     f_train_los.write("\n\n\n")
                # with open(output_acc, 'a') as f_train_los:
                #     f_train_los.write("R\n\n\n")
                #     f_train_los.write(str(acc_list))
                #     f_train_los.write("\n\n\n")

            perf_r = perf_r / R
            perf = perf + perf_r

        perf = perf/K
        res.append(perf)
    print("acc: ",res)