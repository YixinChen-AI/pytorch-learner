import pandas as pd
# 读取训练集
train_df = pd.read_csv('./MNIST_csv/train.csv')
n_train = len(train_df)
n_pixels = len(train_df.columns) - 1
n_class = len(set(train_df['label']))
print('Number of training samples: {0}'.format(n_train))
print('Number of training pixels: {0}'.format(n_pixels))
print('Number of classes: {0}'.format(n_class))

# 读取测试集
test_df = pd.read_csv('./MNIST_csv/test.csv')
n_test = len(test_df)
n_pixels = len(test_df.columns)
print('Number of test samples: {0}'.format(n_test))
print('Number of test pixels: {0}'.format(n_pixels))
# 展示一些图片
import numpy as np
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
random_sel = np.random.randint(len(train_df), size=8)
data = (train_df.iloc[random_sel,1:].values.reshape(-1,1,28,28)/255.)

grid = make_grid(torch.Tensor(data), nrow=8)
plt.rcParams['figure.figsize'] = (16, 2)
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.axis('off')
plt.show()
print(*list(train_df.iloc[random_sel, 0].values), sep = ', ')

# 检查类别是否不均衡
plt.figure(figsize=(8,5))
plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())
plt.xticks(np.arange(n_class))
plt.xlabel('Class', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.grid('on', axis='y')
plt.show()