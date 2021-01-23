import torch
a = torch.tensor([1.,2.])
b = torch.tensor([2.,3.])
#-------------------------
print('减法')
c1 = a - b
c2 = torch.sub(a, b)
print(c1,c2)
#-------------------------
print('乘法')
c1 = a * b
c2 = torch.mul(a, b)
print(c1,c2)
#-------------------------
print('除法')
c1 = a / b
c2 = torch.div(a, b)
print(c1,c2)
#-------------------------
print('加法')
c1 = a + b
c2 = torch.add(a, b)
print(c1,c2)
#-------------------------
print('矩阵乘法')
a = torch.tensor([1.,2.]).view(2,1)
b = torch.tensor([2.,3.]).view(1,2)
print(torch.mm(a, b))
print(torch.matmul(a, b))
print(a @ b)
#-------------------------
print('多维张量矩阵乘法')
a = torch.rand((3,2,64,32))
b = torch.rand((1,2,32,64))
print(torch.matmul(a, b).shape)
#-------------------------
print('幂运算')
a = torch.tensor([1.,2.])
b = torch.tensor([2.,3.])
c1 = a ** b
c2 = torch.pow(a, b)
print(c1,c2)
#-------------------------
print('幂运算')
a = torch.tensor([1.,2.])
b = torch.tensor([2.,3.])
c1 = a ** b
c2 = torch.pow(a, b)
print(c1,c2)
#-------------------------
import numpy as np
print('对数运算')
a = torch.tensor([2,10,np.e])
print(torch.log(a))
print(torch.log2(a))
print(torch.log10(a))
#-------------------------
print('近似运算')
a = torch.tensor(1.2345)
print(a.ceil())
print(a.floor())
print(a.trunc())
print(a.frac())
print(a.round())
#-------------------------
print('剪裁运算')
a = torch.rand(5)
print(a)
print(a.clamp(0.3,0.7))
