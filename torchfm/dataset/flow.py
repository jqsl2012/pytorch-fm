import numpy as np
import pandas as pd
import torch.utils.data


class FlowDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        print('__init___')
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :]
        self.items = data[:, 1:].astype(np.float64) # -1 because ID begins from 1
        self.targets = data[:, 1].astype(np.int)
        #print(self.targets)
        #self.field_dims = np.max(self.items, axis=0) + 1
        self.field_dims = np.array([2, 4, 2, 34, 18, 7, 998, 208, 47, 3, 2, 5, 3, 3, 982, 972, 375, 982, 951, 443, 979, 951, 455, 606, 850, 808, 1012, 573, 573, 87, 670, 588, 664, 663, 670, 99, 98, 52, 1072, 1057, 1059, 390, 1060, 479, 429, 175, 860, 860, 860, 860, 196, 893, 216, 903, 178, 665, 910, 570, 407, 4, 22, 139])

        print(type(self.field_dims), self.field_dims)

    def __len__(self):
        print('.....__len__...', self.targets.shape[0])
        return self.targets.shape[0]

    def __getitem__(self, index):
        #print('.....__getitem__...', index)
        #print(index, self.targets[index], self.items[index])
        return self.items[index], self.targets[index]