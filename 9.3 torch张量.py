import torch
print('tensor的存储区')
a = torch.arange(0,6)
print(a.storage())
b = a.view(2,3)
print(b.storage())
print(id(a)==id(b))
print(id(a.storage)==id(b.storage))
# ----------------------------------
print('研究tensor的切片')
a = torch.arange(0,6)
b = a[2]
print(id(a.storage)==id(b.storage))
# ----------------------------------
print('data_ptr()')
print(a.data_ptr(),b.data_ptr())
print(b.data_ptr()-a.data_ptr())
# ----------------------------------
print('头信息区')
a = torch.arange(0,6)
b = a.view(2,3)
print(a.stride(),b.stride())
print(a.storage_offset(),b.storage_offset())
