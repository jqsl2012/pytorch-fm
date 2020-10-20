import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm
import json

"""
为什么这样做特征处理，参考这篇文章
http://d2l.ai/chapter_recommender-systems/ctr.html
https://www.cnblogs.com/wujianming-110117/p/13224347.html
http://www.ngui.cc/51cto/show-2440.html
"""
class CriteoDataset(torch.utils.data.Dataset):
    """
    Criteo Display Advertising Challenge Dataset

    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
        * Discretize numerical values by log2 transformation which is proposed by the winner of Criteo Competition

    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param min_threshold: infrequent feature threshold.

    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path=None, cache_path='.criteo', rebuild_cache=False, min_threshold=10):
        # self.NUM_FEATS = 39
        # self.NUM_INT_FEATS = 13

        self.NUM_FEATS = 62
        # self.NUM_INT_FEATS = 54
        self.NUM_INT_FEATS = 42

        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        # feat_mapper, defaults = self.__get_feat_mapper(path)

        if 1 == 2:
            feat_mapper, defaults = self.__get_feat_mapper(path)
            np.save('feat_mapper.npy', feat_mapper)
            np.save('defaults.npy', defaults)
            # json_obj = {'feat_mapper': feat_mapper, 'defaults': defaults}
            # json_str = json.dumps(json_obj) + '\n'
            # feat_mapper_file = open('/home/eduapp/pytorch-fm/examples/feat_mapper.txt', 'w')
            # feat_mapper_file.write(json_str)
            # feat_mapper_file.close()

        if 1 == 1:
            feat_mapper = np.load('feat_mapper.npy', allow_pickle=True).item()
            defaults = np.load('defaults.npy', allow_pickle=True).item()
            # print(str(feat_mapper))
            # print(str(defaults))
            print('========={},{}'.format(len(feat_mapper), len(defaults)))
            # with open('/home/eduapp/pytorch-fm/examples/feat_mapper.txt', 'rb') as f:
            #    json_obj = json.load(f)
            #    print(type(json_obj))
            #    feat_mapper, defaults = json_obj['feat_mapper'], json_obj['defaults']

        # feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=1099511627776) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                # print(i, fm)
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        # print('kv=={},{}'.format(key,value))
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: counting features')
            skip = -1
            for line in pbar:
                skip = skip + 1
                if skip == 0:
                    continue
                # values = line.rstrip('\n').split('\t')
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1

        # print('==feat_cnts==')
        # print(feat_cnts)
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}

        # print(feat_mapper)
        # print(defaults)
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            # pbar = tqdm(f, mininterval=1, smoothing=0.1)
            # pbar.set_description('Create criteo dataset cache: setup lmdb')

            skip = -1
            # for line in pbar:
            for line in f:
                skip = skip + 1
                if skip == 0:
                    continue
                # values = line.rstrip('\n').split('\t')
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                # print('============')
                # print(len(feat_mapper))
                # print(feat_mapper.keys())
                for i in range(1, self.NUM_INT_FEATS + 1):
                    # i=str(i)
                    # print(feat_mapper[i], values[i], convert_numeric_feature(values[i]))
                    # print(feat_mapper[i])
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])

                # print('====np_array====')
                # print(np_array)
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = float(val)
    if v > 2:
        return str((math.log(v) ** 2))
    else:
        return str(v - 2)