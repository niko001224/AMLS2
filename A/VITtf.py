#%%
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, ToPILImage, Normalize, Resize
import os

#print(os.getcwd())
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # 加载标注文件S
        img_labels = pd.read_csv(annotations_file)
        
        # 检查图片文件是否存在，并仅保留存在的图片记录
        img_labels['img_path'] = img_labels.iloc[:, 0].apply(lambda x: os.path.join(img_dir, x))
        self.img_labels = img_labels[img_labels['img_path'].apply(os.path.exists)]
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['img_path']
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]  # 假设标签在第二列
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

from torchvision.models import vit_b_16
import torch

# 修改这部分来匹配你的类别数量
num_classes = 5  

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
    model.train()  # 设置模型为训练模式
    total_loss, total_correct, total = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total
    return avg_loss, avg_acc

def validate(dataloader, model, loss_fn):
    model.eval()  # 设置模型为评估模式
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total
    return avg_loss, avg_acc


train_losses, validate_losses, train_accuracies, validate_accuracies = [], [], [], []
transform = Compose([
    Resize((224,224)),
    ToPILImage(),  # 将张量转换为PIL图像
    ToTensor(),  # 再次将PIL图像转换为张量
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CustomImageDataset(
    annotations_file='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train.csv', 
    img_dir='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train_images1', 
    transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

validate_dataset = CustomImageDataset(
    annotations_file='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/train.csv', 
    img_dir='/home/uceexuq/AMLS2-2/cassava-leaf-disease-classification/test_images1', 
    transform=transform)
validate_dataloader = DataLoader(validate_dataset, batch_size=64, shuffle=True)


epochs = 30  # 设置训练的总轮次
for epoch in range(epochs):
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    validate_loss, validate_acc = validate(validate_dataloader, model, loss_fn)
    train_losses.append(train_loss)
    validate_losses.append(validate_loss)
    train_accuracies.append(train_acc)
    validate_accuracies.append(validate_acc)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validate Loss: {validate_loss}, Train Accuracy: {train_acc}, Validate Accuracy: {validate_acc}")



print(train(train_dataloader, model, loss_fn, optimizer))

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(validate_losses, label='Validate Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(validate_accuracies, label='Validate Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%

