# from __future__ import print_function
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# import os,sys,random,time
# import argparse
#
# # from focalloss import *
# from test2.focalloss import FocalLoss
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# start_time = time.time()
# maxe = 0
# for i in range(1000):
#     x = torch.rand(12800,2)*random.randint(1,10)
#     x = Variable(x.to(device))
#     l = torch.rand(12800).ge(0.1).long()
#     l = Variable(l.to(device))
#
#     output0 = FocalLoss(gamma=0)(x,l)
#     output1 = nn.CrossEntropyLoss()(x,l)
#     a = output0.data[0]
#     b = output1.data[0]
#     if abs(a-b)>maxe: maxe = abs(a-b)
# print('time:',time.time()-start_time,'max_error:',maxe)
#
#
# start_time = time.time()
# maxe = 0
# for i in range(100):
#     x = torch.rand(128,1000,8,4)*random.randint(1,10)
#     x = Variable(x.to(device))
#     l = torch.rand(128,8,4)*1000    # 1000 is classes_num
#     l = l.long()
#     l = Variable(l.to(device))
#
#     output0 = FocalLoss(gamma=0)(x,l)
#     output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
#     a = output0.data[0]
#     b = output1.data[0]
#     if abs(a-b)>maxe: maxe = abs(a-b)
# print('time:',time.time()-start_time,'max_error:',maxe)


# from Focal_Loss import focal_loss
import torch

from test2.focalloss import focal_loss

pred = torch.randn((3,5))
print("pred:",pred)

label = torch.tensor([2,3,4])
print("label:",label)


# alpha设定为0.25,对第一类影响进行减弱(目标检测任务中,第一类为背景类)
loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=5)
loss = loss_fn(pred, label)
print(loss)