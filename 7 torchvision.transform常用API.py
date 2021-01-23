from PIL import Image
from torchvision import transforms

def loadImage():
    # 读取图片
    im = Image.open("brunch.jpg")
    im = im.convert("RGB")
    # im.show()
    return im
im = loadImage()

#从中心裁剪一个600*600的图像
output = transforms.CenterCrop(600)(im)
# output.show()
# 从中心裁一个长为600，宽为800的图像
output = transforms.CenterCrop((600,800))(im)
# output.show()
#随机裁剪一个600*600的图像
output = transforms.RandomCrop(600)(im)
# output.show()
#随机裁剪一个600*800的图像
output = transforms.RandomCrop((600,800))(im)
# output.show()
#从上、下、左、右、中心各裁一个300*300的图像
outputs = transforms.FiveCrop(600)(im)
# outputs[4].show()
#p默认为0.5，这里设成1，那么就肯定会水平翻转
output = transforms.RandomHorizontalFlip(p=1.0)(im)
# output.show()
output = transforms.RandomVerticalFlip(p=1)(im)
# output.show()
#在（-30,30）之间选择一个角度进行旋转
output = transforms.RandomRotation(30)(im)
# output.show()
#在60-90之间选择一个角度进行旋转
output = transforms.RandomRotation((60,90))(im)
# output.show()
output = transforms.Resize((400,500))(im)
output.show()

trans = transforms.Compose([transforms.CenterCrop(600),
                            transforms.RandomRotation(30),
                            ])
output = trans(im)
output.show()




