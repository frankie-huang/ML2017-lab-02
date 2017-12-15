import numpy as np
import torch
import random
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据预处理
dtype = torch.FloatTensor

mem = Memory("./mycache")
@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]

x_train, y_train = get_data("a9a")
x_test, y_test = get_data("a9a.t")

# 32561 x 123
x_train = x_train.toarray()
x_train = torch.from_numpy(x_train).type(dtype)

bias = torch.ones(x_train.size(0), 1)
x_train = torch.cat((x_train, bias), 1)

# 32561 x 1
y_train = np.array(y_train).reshape(-1,1)
y_train = torch.from_numpy(y_train).type(dtype)

# 16281 x 122
x_test = x_test.toarray()
x_test = torch.from_numpy(x_test).type(dtype)

# 16281 x 1
y_test = np.array(y_test).reshape(-1,1)
y_test = torch.from_numpy(y_test).type(dtype)

dim = max(x_train.size(1)-1, x_test.size(1))
x_test = torch.cat((x_test, torch.zeros(x_test.size(0), dim - x_test.size(1))), 1) # 补0

w = torch.ones(1, x_train.size(1))
threshold = 0.0

h = torch.mm(x_train, w.t())
y_pred = torch.ones((x_train.size(0),1))
y_pred[h < threshold] = -1

print(torch.mean((y_train == y_train).type(dtype)))
