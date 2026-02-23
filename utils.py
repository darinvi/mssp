import pandas as pd
import numpy as np
import torch

def prepare_spy():
    df = pd.read_csv('spy_raw.csv')[['adjClose']]
    df['adjClose_log'] = np.log(df['adjClose'])
    df['diff_log'] = df['adjClose_log'].diff()
    df = df[['adjClose', 'diff_log']]
    for i in range(1,8):
        df[f'shift_{i}'] = df['diff_log'].shift(i)
    for i in range(3, 9):
        df[f'rolling_{2**i}'] = df['diff_log'].rolling(2**i).mean()
    df['target'] = df['diff_log'].shift(-1)
    df.dropna(inplace=True)
    m = df.min().abs()
    m.iloc[0] = 0
    df = df + m + 1e-6
    df.to_csv('spy.csv', index=False)

def get_spy():
    data = pd.read_csv("spy.csv")
    
    X = torch.tensor(data.drop(columns=['adjClose', 'target']).values, dtype=torch.double)
    y = torch.tensor(data['target'].values, dtype=torch.double)

    X_train = X[:int(len(X)*0.8)]
    y_train = y[:int(len(X)*0.8)]
    close_train = data['adjClose'][:int(len(X)*0.8)]

    X_valid = X[int(len(X)*0.8):int(len(X)*0.9)]
    y_valid = y[int(len(X)*0.8):int(len(X)*0.9)]
    close_valid = data['adjClose'][int(len(X)*0.8):int(len(X)*0.9)]
    
    X_test = X[int(len(X)*0.9):]
    y_test = y[int(len(X)*0.9):]
    close_test = data['adjClose'][int(len(X)*0.9):]

    return X_train, y_train, torch.tensor(close_train.values), X_valid, y_valid, torch.tensor(close_valid.values), X_test, y_test, torch.tensor(close_test.values)

def inverse_transform_spy(pred, close):
    return np.exp(pred - torch.tensor(0.115887)) * close

