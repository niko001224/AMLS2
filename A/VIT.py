import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, ToPILImage, Normalize, Resize
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

from torchvision.models import vit_b_16
import torch

# 修改这部分来匹配你的类别数量
num_classes = 5  # 假设有10个类别

model = vit_b_16(pretrained=True)
model.heads[0] = torch.nn.Linear(model.heads[0].in_features, num_classes)

# 移动模型到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model.to(device)

from torch import nn, optim

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

transform = Compose([
    Resize((224,224)),
    ToPILImage(),  # 将张量转换为PIL图像
    # 这里可以插入一些PIL图像的转换，例如 Resize 等
    ToTensor(),  # 再次将PIL图像转换为张量
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = CustomImageDataset(annotations_file='./cassava-leaf-disease-classification/train.csv', img_dir='./cassava-leaf-disease-classification/images', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

train(train_dataloader, model, loss_fn, optimizer)
