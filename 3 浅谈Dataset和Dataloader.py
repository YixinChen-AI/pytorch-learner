import torch
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
        self.label = torch.LongTensor([1,1,0,0])

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)
#--------------------------------------------
print('第一部分')
mydataset = MyDataset()
mydataloader = DataLoader(dataset=mydataset,
                          batch_size=1)

for i,(data,label) in enumerate(mydataloader):
    print(data,label)
#--------------------------------------------
print('第二部分')
mydataloader = DataLoader(dataset=mydataset,
                          batch_size=2,
                          shuffle=True)

for i,(data,label) in enumerate(mydataloader):
    print(data,label)
#--------------------------------------------
print('第三部分')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for i,(data,label) in enumerate(mydataloader):
    data = data.to(device)
    label = label.to(device)
    print(data,label)