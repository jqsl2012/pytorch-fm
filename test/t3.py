import torch
import random

n_train = 22#len(10)
split = n_train // 3
print(split)
indices = random.shuffle(list(range(n_train)))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
# train_loader = DataLoader(..., sampler=train_sampler, ...)
# valid_loader = DataLoader(..., sampler=valid_sampler, ...)