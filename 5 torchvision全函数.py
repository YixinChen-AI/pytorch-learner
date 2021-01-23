import torchvision
import torch
import torchvision.models as models

mydataset = torchvision.datasets.MNIST(root='./',
                                      train=True,
                                      transform=None,
                                      target_transform=None,
                                      download=True)

resnet18 = models.resnet18(pretrained=True)
print(resnet18.layer3)
exit()
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)