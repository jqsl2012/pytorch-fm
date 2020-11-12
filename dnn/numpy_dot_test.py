import numpy as np

"""
https://www.cnblogs.com/luhuan/p/7925790.html
"""
x = np.array([[1, 2], [3, 4]])
print(x)
print(type(x))

h = np.array([[1, 2, 3], [4, 5, 6]])
print(h)
# dot：第一个矩阵的第一行元素 与 第二个矩阵的第一列元素，两两相乘，然后求和，得到的元素作为最终矩阵的对应行列的元素值。
# 这里是：
"""
[1,2]*[1,4] => 1*1+2*4 => 9    然后[1,2]的行索引是0,[1,4]的列索引是0，所以9填充到(0,0)索引位置
[3,4]*[3,6] => 3*3+4*6 => 33   然后[3,4]是行索引1，[3,6]是列索引2，所以33填充到(1,2)索引位置
"""
v = x.dot(h)
print(v)
print(type(v))