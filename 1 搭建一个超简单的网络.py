import torch
import torch.nn as nn
import numpy as np

# 构建输入集
x = np.mat('0 0;'
           '0 1;'
           '1 0;'
           '1 1')
x = torch.tensor(x).float()
y = np.mat('1;'
           '0;'
           '0;'
           '1')
y = torch.tensor(y).float()

myNet = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,1),
    nn.Sigmoid()
    )
print(myNet)

optimzer = torch.optim.SGD(myNet.parameters(),lr=0.05)
loss_func = nn.MSELoss()

for epoch in range(5000):
    out = myNet(x)
    loss = loss_func(out,y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

print(myNet(x).data)