import torch
from torch import nn
from torch.autograd import Variable
# 定义词嵌入
embeds = nn.Embedding(2, 5) # 2 个单词，维度 5
# 得到词嵌入矩阵,开始是随机初始化的
torch.manual_seed(1)
print(embeds.weight)
print(embeds)


# 访问第 50 个词的词向量
# embeds = nn.Embedding(100, 10)
print(embeds(Variable(torch.LongTensor([1]))))   # 访问第2个词的词向量
# print(embeds(0))  # 报错
print(embeds(Variable(torch.LongTensor([1,0]))))  # 访问第1和2个词的词向量
print('读取多个向量: ')
# print(embeds(Variable(torch.LongTensor(2,5))))  #
# print(torch.LongTensor(3,4))
# print(Variable(torch.LongTensor([1,2,3])))

# unsqueeze(0)
# a = torch.randn(1,3)
# print('unsqueeze(0): ', a, a.unsqueeze(0))
# print(torch.sum(a))

# print('=======================')
# embedding = nn.Embedding(10, 3)
# print(embedding.weight)
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# print(embedding(input))