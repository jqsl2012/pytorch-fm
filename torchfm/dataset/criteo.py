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

class CriteoDataset(torch.utils.data.Dataset):
    """
    Criteo Display Advertising Challenge Dataset

    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
        * Discretize numerical values by log2 transformation which is proposed by the winner of Criteo Competition

    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param min_threshold: infrequent feature threshold.  罕见特征阈值，小于min_threshold的则用原始特征值，否则转化成索引，然后用词嵌入

    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path=None, cache_path='.criteo', rebuild_cache=False, min_threshold=10):
        # self.NUM_FEATS = 39
        # self.NUM_INT_FEATS = 13
        self.NUM_FEATS = 62
        self.NUM_INT_FEATS = 54

        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1     # 统计总训练记录数
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)    # 统计每个字段的维度

    # 返回一条数据或一个样本
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]    # 特征,标签

    # 返回样本数量
    def __len__(self):
        return self.length

    # 把训练数据存储到lmdb这个持久化缓存中
    def __build_cache(self, path, cache_path):
        # feat_mapper, defaults存储成json写入文件
        if True:
            feat_mapper, defaults = self.__get_feat_mapper(path)

            np.save('feat_mapper.npy', feat_mapper)
            np.save('defaults.npy', defaults)

            # json_obj = {'feat_mapper': feat_mapper, 'defaults': defaults}
            # json_str = json.dumps(json_obj) + '\n'
            # feat_mapper_file = open('/home/eduapp/pytorch-fm/examples/feat_mapper.txt', 'w')
            # feat_mapper_file.write(json_str)
            # feat_mapper_file.close()

        if True:
            feat_mapper = np.load('feat_mapper.npy', allow_pickle=True).item()
            defaults = np.load('defaults.npy', allow_pickle=True).item()
            # feat_mapper_file = open('/home/eduapp/pytorch-fm/examples/feat_mapper.txt', 'r', encoding='utf-8')
            # lines = feat_mapper_file.readlines()
            # json_obj = json.loads(lines[0].strip('\n'))
            # with open('/home/eduapp/pytorch-fm/examples/feat_mapper.txt', 'rb') as f:
            #     json_obj = json.load(f)
            #     print(type(json_obj))
            # feat_mapper, defaults = json_obj['feat_mapper'], json_obj['defaults']
            # feat_mapper_file.close()


        with lmdb.open(cache_path, map_size=1e11) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[int(i) - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        # print('kv=={},{}'.format(key,value))
                        txn.put(key, value)

    # 获取features mapping，特征映射关系
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
                for i in range(1, self.NUM_INT_FEATS + 1):  # 读取稀疏特征
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1): # 读取稠密特征
                    feat_cnts[i][values[i]] += 1

        # print('==feat_cnts==')
        # print(feat_cnts)
        #i是特征索引，feat是特征的值
        """
        {1: {'-2.0': 0}, 2: {}, 3: {'-2.0': 0}, 4: {}, 5: {'-2.0': 0}, 6: {'-2.0': 0}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {'-2.0': 0}, 12: {}, 13: {'-2.0': 0}, 14: {}, 15: {}, 16: {}, 17: {}, 18: {}, 19: {}, 20: {}, 21: {}, 22: {}, 23: {}, 24: {}, 25: {}, 26: {}, 27: {}, 28: {}, 29: {}, 30: {}, 31: {}, 32: {}, 33: {}, 34: {}, 35: {}, 36: {}, 37: {}, 38: {'-2.0': 0}, 39: {}, 40: {}, 41: {}, 42: {}, 43: {}, 44: {}, 45: {}, 46: {}, 47: {}, 48: {}, 49: {}, 50: {}, 51: {}, 52: {}, 53: {}, 54: {}, 55: {}, 56: {}, 57: {}, 58: {}, 59: {'5595': 0}, 60: {}, 61: {}, 62: {}}
        {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 1, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 1, 60: 0, 61: 0, 62: 0}
        第一个for循环for i, cnt in feat_cnts.items()，是循环处理每一个特征，
        第二个for是循环处理每个特征下的所有值和此值出现的次数。
        {特征序号: {'特征值': 次数}} 这样的形式为feat_mapper特征映射关系
        """
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    # 1e11 <=> 1099511627776
    # 读取实际的训练数据文件，转化成字节数据后返回
    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: setup lmdb')
            skip = -1
            for line in pbar:
                skip = skip+1
                if skip == 0:
                    continue
                # values = line.rstrip('\n').split('\t')
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])    # 第一列的label
                for i in range(1, self.NUM_INT_FEATS + 1):
                    # TODO 这里是把原始特征值转换成索引吗？ 如果转换不了则使用原始默认值
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    # TODO 这里同上
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])

                #TODO 这里为什么要得到所有特征值的索引呢？np_array是索引。item_idx是每行数据自增？为啥要自增？
                # 用自增是因为每行数据以自增id存储到缓存
                # 稀疏和稠密特征转换成索引np_array，是为了做词嵌入，词嵌入能够做到特征的高维组合，扔到DNN中进行全连接训练，
                # 而这是LR,FM等做不到的。
                # 既然知道了训练时候特征的转换为索引，为词嵌入做准备，那么我们新来的数据也要做类似的事情，才能用训练好的模型进行预测
                # 否则你拿另一份数据来预测是会有问题的。
                """
                np_array = [  0   0   1   0   9  10   1 457 185  41   1   0   2   1   0 873 369 235
                             580 822 265 679 506 144  44 140 682 849 135 135  84  28 392 126 114 346
                              95  91  43 317 554 407 148 400 418 232 161 190 245 629 254  75 793   7
                             292  12 664 482 127  63   0   0  25]
                 np_array[0]=label
                 np_array[1:]=所有特征值的索引
                 
                 新来的数据包括未见过的测试数据是否是在这里转化为索引np_array？
                 且使用上面老的数据生成的（feat_mapper, defaults）这个是老数据生成的，新数据只用，不改变他们，
                 如果改变了，则预测就不准的了。。。
                 1. 老数据生成的feat_mapper, defaults得先存起来
                 2. 新来的数据使用，那么新来的数据的特征就转换成了np_array
                """
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer


# 稠密特征被这样处理了。
@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = float(val)
    if v > 2:
        # return str(int(math.log(v) ** 2))
        return str((math.log(v) ** 2))
    else:
        return str(v - 2)
