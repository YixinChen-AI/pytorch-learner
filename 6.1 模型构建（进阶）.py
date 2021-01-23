import torch.nn as nn
import torch

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,64,3)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
net = MyNet()
print(net)
#--------------------------------------
print('------------------------------')
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.add_module('conv1',nn.Conv2d(3,64,3))
        self.add_module('conv2',nn.Conv2d(64,64,3))

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
net =  MyNet()
print(net)
#--------------------------------------
print('------------------------------')
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.linears = [nn.Linear(10,10) for i in range(5)]

    def forward(self,x):
        for l in self.linears:
            x = l(x)
        return x
net = MyNet()
print(net)
#--------------------------------------
print('------------------------------')
vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class Model1(nn.Module):
    def __init__(self):
        super(Model1,self).__init__()

        self.vgg = nn.ModuleList(vgg(vgg_cfg,3))

    def forward(self,x):

        for l in self.vgg:
            x = l(x)
m1 = Model1()
print(m1)

#--------------------------------------
print('------------------------------')
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.Conv2d(64,64,3)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
net = MyNet()
print(net)

#--------------------------------------
print('------------------------------')
from collections import OrderedDict
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(3,64,3)),
            ('conv2',nn.Conv2d(64,64,3))
        ]))

    def forward(self,x):
        x = self.conv(x)
        return x
net = MyNet()
print(net)