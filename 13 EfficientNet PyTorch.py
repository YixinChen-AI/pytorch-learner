from efficientnet_pytorch import EfficientNet
import efficientnet_pytorch
import torch
model = EfficientNet.from_name('efficientnet-b0')
stage1 = [];stage2 = [];stage3 = [];stage4 = [];stem=[]
# for i,module in enumerate(model.modules()):
#     if i in [1,2,3]:
#         stem.append(module)
#     if i in [5,17,32]:
#         stage1.append(module)
#     if i in [47,62]:
#         stage2.append(module)
#     if i in [77,92,107,122,137,152]:
#         stage3.append(module)
#     if i in [167,182,197,212,227]:
#         stage4.append(module)
# stem = torch.nn.Sequential(*stem)
# stage1 = torch.nn.Sequential(*stage1)
# stage2 = torch.nn.Sequential(*stage2)
# stage3 = torch.nn.Sequential(*stage3)
# stage4 = torch.nn.Sequential(*stage4)
#
# for module in stage1.modules():
#     if isinstance(module,torch.nn.BatchNorm2d):
#         stage1_out_channels = module.weight.shape[0]
# for module in stage2.modules():
#     if isinstance(module,torch.nn.BatchNorm2d):
#         stage2_out_channels = module.weight.shape[0]
# for module in stage3.modules():
#     if isinstance(module,torch.nn.BatchNorm2d):
#         stage3_out_channels = module.weight.shape[0]
# for module in stage4.modules():
#     if isinstance(module,torch.nn.BatchNorm2d):
#         stage4_out_channels = module.weight.shape[0]
stage = [[],[],[],[],[],[]];ind = 0
for i,module in enumerate(model.modules()):
    if isinstance(module,efficientnet_pytorch.model.MBConvBlock):
        if ind == 0:
            ind += 1
        if module._depthwise_conv.stride[0] == 2:
            ind += 1
            stage[ind].append(module)
        elif module._depthwise_conv.stride[0] == 1:
            stage[ind].append(module)
    elif i in [1,2,3]:
        stage[0].append(module)
stem = torch.nn.Sequential(*stage[0])
stage1 = torch.nn.Sequential(*stage[1])
stage2 = torch.nn.Sequential(*stage[2])
stage3 = torch.nn.Sequential(*stage[3])
stage4 = torch.nn.Sequential(*stage[4])
stage5 = torch.nn.Sequential(*stage[5])
for module in stage1.modules():
    if isinstance(module,torch.nn.BatchNorm2d):
        stage1_out_channels = module.weight.shape[0]
for module in stage2.modules():
    if isinstance(module,torch.nn.BatchNorm2d):
        stage2_out_channels = module.weight.shape[0]
for module in stage3.modules():
    if isinstance(module,torch.nn.BatchNorm2d):
        stage3_out_channels = module.weight.shape[0]
for module in stage4.modules():
    if isinstance(module,torch.nn.BatchNorm2d):
        stage4_out_channels = module.weight.shape[0]
for module in stage5.modules():
    if isinstance(module,torch.nn.BatchNorm2d):
        stage5_out_channels = module.weight.shape[0]

x1 = stem(torch.rand(1,3,1000,1000))
x1 = stage1(x1)

print(x1.shape)
exit()
print(stage1)
print(stage2)
print(stage3)
print(stage4)
print(stage5)
