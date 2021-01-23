import torch
import numpy as np
#---------------------------
print('numpy 和torch互相转换')
a = np.array([1.,2.,3.])
b = torch.tensor(a)
c = b.numpy()
print(a)
print(b)
print(c)
#---------------------------
print('numpy 和torch互相转换1')
a = np.array([1,2,3],dtype=np.float64)
b = torch.Tensor(a)
b[0] = 999
print('共享内存' if a[0]==b[0] else '不共享内存')
#---------------------------
print('numpy 和torch互相转换2')
a = np.array([1,2,3],dtype=np.float32)
b = torch.Tensor(a)
b[0] = 999
print('共享内存' if a[0]==b[0] else '不共享内存')

#---------------------------
print('from_numpy()')
a = np.array([1,2,3],dtype=np.float64)
b = torch.from_numpy(a)
print('共享内存' if a[0]==b[0] else '不共享内存')
a = np.array([1,2,3],dtype=np.float32)
b = torch.from_numpy(a)
b[0] = 999
print('共享内存' if a[0]==b[0] else '不共享内存')

#---------------------------
print('numpy()与numpy().copy()')
b = torch.tensor([1.,2.,3.])
a = b.numpy()
a[0]=999
print(a,b)
print('共享内存' if a[0]==b[0] else '不共享内存')
b = torch.tensor([1.,2.,3.])
a = b.numpy().copy()
a[0]=999
print(a,b)
print('共享内存' if a[0]==b[0] else '不共享内存')

#---------------------------
print('命名规则')
a = torch.rand(2,3,4)
b = np.random.rand(2,3,4)

#---------------------------
print('张量分割')
a = torch.rand(2,3,4)
b = a.numpy()
pytorch_masked = a[a > 0.5]
numpy_masked = b[b > 0.5]
print(pytorch_masked)
print(numpy_masked)