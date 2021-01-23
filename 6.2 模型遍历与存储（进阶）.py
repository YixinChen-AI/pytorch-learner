import torch.nn as nn
import torch
from collections import OrderedDict
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2 = nn.Conv2d(64,64,3)
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.features = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(64,128,3)),
            ('conv4', nn.Conv2d(128,128,3)),
            ('relu1', nn.ReLU())
        ]))

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.features(x)

        return x
net = MyNet()
print(net)

#--------------------------------------
print('-------------------------------')
for idx,m in enumerate(net.modules()):
    print(idx,"-",m)
#--------------------------------------
print('-------------------------------')
for idx,(name,m) in enumerate(net.named_modules()):
    print(idx,"-",name)

#--------------------------------------
print('-------------------------------')
for p in net.parameters():
    print(type(p.data),p.size())
#--------------------------------------
print('-------------------------------')
for idx,(name,m) in enumerate(net.named_parameters()):
    print(idx,"-",name,m.size())
#--------------------------------------
print('-------------------------------')
for idx,(name,m) in enumerate(net.named_modules()):
    if isinstance(m,nn.Conv2d):
        print(m.weight.shape)
        print(m.bias.shape)
#--------------------------------------
# print('-------------------------------')
# torch.save(net,'model.pth') # 保存
# net = torch.load("model.pth") # 加载