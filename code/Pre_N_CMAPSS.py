import os
import numpy as np
import pandas as pd
import torch
import h5py
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

np.set_printoptions(threshold=np.inf)

seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


TIME_WINDOW = 60          # 25,30,35,40,45,50....


def dataloader(filename, save_description=False, print_description=False):
    dirname = "../dataset/N-CMAPSS"
    filepath = os.path.normpath(os.path.join(os.path.join(os.getcwd(), dirname), filename))

    with h5py.File(filepath, "r") as hdf:
        print("Keys: {}".format(hdf.keys()))
        # Development set
        W_dev = np.array(hdf.get('W_dev'))  # W  - operative condition
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s - measured signal
        X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v - virtual sensors
        T_dev = np.array(hdf.get('T_dev'))  # T - engine health parameters
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL - RUL label
        A_dev = np.array(hdf.get(
            'A_dev'))  # Auxiliary - unit number u and the flight cycle number c, the flight class Fc and the health state h s

        # Test set
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        X_v_test = np.array(hdf.get('X_v_test'))  # X_v
        T_test = np.array(hdf.get('T_test'))  # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        X_v_var = list(np.array(X_v_var, dtype='U20'))
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))

    if print_description:
        W = np.concatenate((W_dev, W_test), axis=0)
        X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
        X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
        T = np.concatenate((T_dev, T_test), axis=0)
        Y = np.concatenate((Y_dev, Y_test), axis=0)
        A = np.concatenate((A_dev, A_test), axis=0)

        print("W shape: " + str(W.shape))
        print("X_s shape: " + str(X_s.shape))
        print("X_v shape: " + str(X_v.shape))
        print("T shape: " + str(T.shape))
        print("Y shape: " + str(Y.shape))
        print("A shape: " + str(A.shape))
        print("Variables in W_var: {}".format(W_var))
        print("Variables in X_s_var: {}".format(X_s_var))
        print("Variables in X_v_var: {}".format(X_v_var))
        print("Variables in T_var: {}".format(T_var))
        print("Variables in A_var: {}".format(A_var))

    dev_data = np.concatenate((W_dev, X_s_dev, X_v_dev, T_dev, A_dev, Y_dev), axis=1)
    test_data = np.concatenate((W_test, X_s_test, X_v_test, T_test, A_test, Y_test), axis=1)
    column_name = W_var + X_s_var + X_v_var + T_var + A_var
    column_name.append("RUL")

    if print_description:
        print("dev_data shape: {}".format(dev_data.shape))
        print("test_data shape: {}".format(test_data.shape))
        print("column_name shape: {}".format(len(column_name)))
        print("column_name: {}".format(column_name))

    df_dev = pd.DataFrame(data=dev_data, columns=column_name)
    df_test = pd.DataFrame(data=test_data, columns=column_name)

    if print_description and save_description:
        with open("dataset_info.txt", "w+") as f:
            f.write("\ndf_dev shape: {}".format(df_dev.shape))
            f.write("\ndf_test shape: {}".format(df_test.shape))
            f.write("\ncolumn_name: {}".format(column_name))
            f.write("\nVariables in W_var: {}".format(W_var))
            f.write("\nVariables in X_s_var: {}".format(X_s_var))
            f.write("\nVariables in X_v_var: {}".format(X_v_var))
            f.write("\nVariables in T_var: {}".format(T_var))
            f.write("\nVariables in A_var: {}".format(A_var))

    return df_dev, df_test


def Z_score_normalization(dataframe):

    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.drop('unit')

    for col in numeric_cols:
        group_means = dataframe.groupby('unit')[col].transform('mean')
        group_stds = dataframe.groupby('unit')[col].transform('std')
        group_stds = group_stds.replace(0, 1)
        dataframe[col] = (dataframe[col] - group_means) / group_stds

    return dataframe


def data_generate(filename, add_lag=False):
    # Pre-processing based on EDA findings
    df_dev, df_test = dataloader(filename)
    df_dev = df_dev.drop(
        columns=[ "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
                 "HPT_flow_mod", "cycle",
                 "Fc", "W21", "W22", "W25", "W31", "W32", "W48", "W50", "P30", "P45",
                 "hs","alt","Mach","TRA","T2","T40",
                 "SmFan", "SmLPC", "SmHPC", "phi"])
    df_test = df_test.drop(
        columns=[ "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
                 "HPT_flow_mod", "cycle",
                 "Fc", "W21", "W22", "W25", "W31", "W32", "W48", "W50", "P30", "P45",
                 "hs","alt","Mach","TRA","T2","T40",
                 "SmFan", "SmLPC", "SmHPC", "phi"])


    if add_lag:
        df_dev["RUL_lag1"] = df_dev["RUL"].shift(1)
        df_dev["RUL_lag3"] = df_dev["RUL"].shift(3)
        df_dev["RUL_lag5"] = df_dev["RUL"].shift(5)
        df_dev = df_dev.iloc[5::]  # Discard NaN rows

    # Model training
    X_train = df_dev.drop(["RUL"], axis=1)
    Y_train = df_dev["RUL"]
    X_test = df_test.drop(["RUL"], axis=1)
    Y_test = df_test["RUL"]

    # normalize input features
    X_train = Z_score_normalization(X_train)
    X_test = Z_score_normalization(X_test)

    return X_train, Y_train, X_test, Y_test


class MyDataG(Dataset):
    def __init__(self, x_train, y_train, step_size=TIME_WINDOW):

        self.step_size = step_size
        self.x_data = torch.from_numpy(x_train.drop(["unit"], axis=1).to_numpy())
        self.y_data = torch.from_numpy(y_train.to_numpy())
        self.ids = torch.from_numpy(x_train["unit"].to_numpy())

        self.samples_x = []
        self.samples_y = []

        unique_ids = torch.unique(self.ids)

        for engine_id in unique_ids:
            engine_indices = (self.ids == engine_id).nonzero(as_tuple=True)[0]
            engine_x = self.x_data[engine_indices]
            engine_y = self.y_data[engine_indices]

            for i in range(0, engine_x.shape[0] - TIME_WINDOW + 1, step_size):
                window_x = engine_x[i:i+TIME_WINDOW]
                label_y = engine_y[i + TIME_WINDOW - 1].unsqueeze(0)
                label_y = torch.clamp(label_y, max=65)

                self.samples_x.append(window_x.unsqueeze(-1))
                self.samples_y.append(label_y)

        self.len = len(self.samples_x)

    def __getitem__(self, index):
        return self.samples_x[index], self.samples_y[index]

    def __len__(self):
        return self.len



class MyDataG2(Dataset):
    def __init__(self,x_test, y_test ):
        self.x_data = torch.from_numpy(x_test.drop(["unit"], axis=1).to_numpy())
        self.y_data = torch.from_numpy(y_test.to_numpy())
        self.ids = torch.from_numpy(x_test["unit"].to_numpy())

        self.samples_x = []
        self.samples_y = []

        unique_ids = torch.unique(self.ids)

        for engine_id in unique_ids:
            engine_indices = (self.ids == engine_id).nonzero(as_tuple=True)[0]
            engine_x = self.x_data[engine_indices]
            engine_y = self.y_data[engine_indices]

            if engine_x.shape[0] >= TIME_WINDOW:
                start_idx = engine_x.shape[0] - TIME_WINDOW
                window_x = engine_x[start_idx:start_idx + TIME_WINDOW]
                label_y = engine_y[start_idx + TIME_WINDOW - 1]

                self.samples_x.append(window_x.unsqueeze(-1))
                self.samples_y.append(label_y)

        self.len = len(self.samples_x)

    def __getitem__(self, index):
        return self.samples_x[index], self.samples_y[index]

    def __len__(self):
        return self.len
