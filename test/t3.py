import torch
import random

# n_train = 22#len(10)
# split = n_train // 3
# print(split)
# indices = random.shuffle(list(range(n_train)))
# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
# valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
# # train_loader = DataLoader(..., sampler=train_sampler, ...)
# # valid_loader = DataLoader(..., sampler=valid_sampler, ...)

print('a=', 1)

from sklearn import preprocessing
from pickle import dump, dumps
from pickle import load
le = preprocessing.LabelEncoder()
print(le.fit(["paris", "paris", "tokyo", "amsterdam"]))
print(le.transform(["tokyo", "tokyo", "paris"]))
print(list(le.inverse_transform([2, 2, 1])))

print(le.fit_transform(['a', 'b', 'c', 'd']))

# save the scaler
dump(le, open('LabelEncoder.pkl', 'wb'))

le = load(open('LabelEncoder.pkl', 'rb'))
# print(le.fit(['c', 'd', 'e']))
print(le.transform(['c', 'd'])) # 正确写法

dic = dict(zip(le.classes_, le.transform(le.classes_)))
print(dic)

print(le.transform(['c', 'd', 'e', 'f'])) # 正确写法
# print(le.fit_transform(['c', 'd'])) # 错误写法，fit会导致已sabe的preprocessing被初始化