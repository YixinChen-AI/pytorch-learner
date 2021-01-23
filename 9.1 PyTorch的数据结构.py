import torch
import numpy as np
#----------------------
print('torch的浮点数与整数的默认数据类型')
a = torch.tensor([1,2,3])
b = torch.tensor([1.,2.,3.])
print(a,a.dtype)
print(b,b.dtype)

#----------------------
print('torch的浮点数与整数的默认数据类型')
a = torch.tensor([1,2,3],dtype=torch.int8)
b = torch.tensor([1.,2.,3.],dtype = torch.float64)
print(a,a.dtype)
print(b,b.dtype)

#----------------------
print('torch的浮点数与numpy的浮点数的默认数据类型')
a = torch.tensor([1.,2.,3.])
b = np.array([1.,2.,3.])
print(a,a.dtype)
print(b,b.dtype)

#-----------------------
print('torch的构造函数')
a = torch.IntTensor([1,2,3])
b = torch.LongTensor([1,2,3])
c = torch.FloatTensor([1,2,3])
d = torch.DoubleTensor([1,2,3])
e = torch.tensor([1,2,3])
f = torch.tensor([1.,2.,3.])
print(a.dtype)
print(b.dtype)
print(c.dtype)
print(d.dtype)
print(e.dtype)
print(f.dtype)

#-----------------------
print('数据类型转换1')
a = torch.tensor([1,2,3])
b = a.float()
c = a.double()
d = a.long()
print(b.dtype)
print(c.dtype)
print(d.dtype)
print('数据类型转换2')
b = a.type(torch.float32)
c = a.type(torch.float64)
d = a.type(torch.int64)
print(b.dtype)
print(c.dtype)
print(d.dtype)




