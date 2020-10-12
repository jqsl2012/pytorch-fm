# import torch
# import random
#
# n_train = 22#len(10)
# split = n_train // 3
# print(split)
# indices = random.shuffle(list(range(n_train)))
# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
# valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
# # train_loader = DataLoader(..., sampler=train_sampler, ...)
# # valid_loader = DataLoader(..., sampler=valid_sampler, ...)
from collections import defaultdict

if 1==2:
    import random as rd

    for i in range(100):
        label = 1 if rd.random() >= 0.5 else 0

        line = 'feat_idx:1 '
        # line = line + ' feat_idx:1 '
        # line = line + 'feat_value:' + str(rd.random()) + ' '
        line = line + 'feat_value:' + str(rd.random()) + ' '
        line = line + 'label:' + str(label)
        print(line)

if 1==1:
    # arr = []
    feat_cnts = defaultdict(lambda: defaultdict(int))

    # arr[0][0] = 1
    feat_cnts[13][1] = 1
    # feat_cnts[0].append(2)
    print(feat_cnts)
    print(feat_cnts.items())