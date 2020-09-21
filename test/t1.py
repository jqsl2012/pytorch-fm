import torch

field_dims = [1, 2]

# print(sum(field_dims))


fc = torch.nn.Embedding(3, 2)
print(fc)

x = torch.LongTensor([[1, 2], [2, 1]])
print(fc(x))

# x = x + x.new_tensor(self.offsets).unsqueeze(0)
s = torch.sum(fc(x), dim=1)
print(s)

print('sigmoid_v ...')
sigmoid_v = torch.sigmoid(s.squeeze(1))
print(sigmoid_v)
