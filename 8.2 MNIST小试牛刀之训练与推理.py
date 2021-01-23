import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

train_df = pd.read_csv('./MNIST_csv/train.csv')
test_df = pd.read_csv('./MNIST_csv/test.csv')
n_train = len(train_df)
n_test = len(test_df)
n_pixels = len(train_df.columns) - 1
n_class = len(set(train_df['label']))



class MNIST_data(Dataset):
    def __init__(self, file_path,
                 transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))])
                 ):
        df = pd.read_csv(file_path)
        if len(df.columns) == n_pixels:
            # test data
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:,:,:,None]
            self.y = None
        else:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])


batch_size = 64

train_dataset = MNIST_data('./MNIST_csv/train.csv',
                           transform= transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Grayscale(num_output_channels=3),
                            transforms.RandomRotation(degrees=20),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_dataset = MNIST_data('./MNIST_csv/test.csv',
                          transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Grayscale(num_output_channels=3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=(0.5,), std=(0.5,))]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.features = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features1(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
# model = torchvision.models.resnet50(pretrained=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss().to(device)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
print(model)


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # 读入数据
        data = data.to(device)
        target = target.to(device)
        # 计算模型预测结果和损失
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad() # 计算图梯度清零
        loss.backward() # 损失反向传播
        optimizer.step()# 然后更新参数
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
            log.append(loss.item())
    exp_lr_scheduler.step()

log = []
n_epochs = 10
for epoch in range(n_epochs):
    train(epoch)

plt.plot(log)
plt.show()
# 保存（载入）模型
torch.save(model,'./MNIST_csv/model.pth')
model = torch.load('./MNIST_csv/model.pth')

def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        data = data.to(device)
        output = model(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    return test_pred

test_pred = prediciton(test_loader)

from torchvision.utils import make_grid
random_sel = np.random.randint(len(test_df), size=8)
data = (test_df.iloc[random_sel,:].values.reshape(-1,1,28,28)/255.)

grid = make_grid(torch.Tensor(data), nrow=8)
plt.rcParams['figure.figsize'] = (16, 2)
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.axis('off')
plt.show()
print(*list(test_pred[random_sel].numpy()), sep = ', ')