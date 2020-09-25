import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
import pandas as pd


numDataPoints = 1000
data_dim = 5
bs = 100

# Create dummy data with class imbalance 9 to 1
data = torch.FloatTensor(numDataPoints, data_dim)
target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),
                    np.ones(int(numDataPoints * 0.1), dtype=np.int32)))

target = pd.read_csv('/home/eduapp/pytorch-fm/examples/all_features_use_model_estimate.fe_output.40w.csv', usecols=[0], header=None)
print(target)
print(type(target))
target = target.to_numpy()
# 1/0
print('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
print(samples_weight)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
1/0

target = torch.from_numpy(target).long()
train_dataset = torch.utils.data.TensorDataset(data, target)

train_loader = DataLoader(
    train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

for i, (data, target) in enumerate(train_loader):
    print("batch index {}, 0/1: {}/{}".format(
        i,
        len(np.where(target.numpy() == 0)[0]),
        len(np.where(target.numpy() == 1)[0])))