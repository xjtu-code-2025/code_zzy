import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import models
from torch.optim import SGD

from PIL import Image
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import matplotlib.pyplot as plt

from torch.utils.data import random_split
# root="D:/testPython/暑期课程/pet/images"
# label_list = os.listdir(root)
# label_a = []
# for i in label_list:
#     if i.endswith(("png", "jpg", "gif")):
#         label=i.strip().rsplit("_", 1)[0]#获取Abyssinian
#         print(label)
#         label_a.append(label)#得到label列表
# print(len(labels))
# labels = list(set(labels))
# print(len(labels))
# print(label_list[1:10])

# label_list[3],id= label_list[3].strip().split("_")
# print(label_list[3])

# image_folder = os.path.join(root, label)

class own_dataset(Dataset):
    # init,len,getitem
    def __init__(self, root, preprocess):
        super(own_dataset, self).__init__()
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
    
        label_a = []
        label_list = os.listdir(root)#获得每一张图片的Abyssinian_100.mat
        for i in label_list:
            if i.endswith(("png", "jpg", "gif")):
                label=i.strip().rsplit("_", 1)[0]#获取Abyssinian
                label_a.append(label)#得到label不重复列表，print(len(labels))=37
        label_a = list(set(label_a))
        for j in label_list:
            if j.endswith(("png", "jpg", "gif")):
                label=j.strip().rsplit("_", 1)[0]#获取Abyssinian
                self.image_paths.append(os.path.join(root, j))#得到每张图片路径
                self.labels.append(label_a.index(label))#得到每张图片对应的标签
        print(self.labels)
        # label_list = os.listdir(root)
        # label_temp = []
        # idx = 0
        # for i in label_list:
        #     if i.endswith(("png", "jpg", "gif")):
        #         label=i.strip().rsplit("_", 1)[0]
        #         label_temp.append(label)
        
        # label_set = list(set(label_temp))
        # label_mapping = {}
        # for i in range(len(label_set)):
        #     label_mapping[label_set[i]]=i
        
        # for i in range(len(label_temp)):
        #    label_temp[i] = label_mapping[label_temp[i]]
        # self.labels = label_temp
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.preprocess(image)
        label = self.labels[item]
        return image, label

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
print(args)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(256),          # 短边缩放到256
    transforms.CenterCrop(224),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),          # 短边缩放到256
    transforms.CenterCrop(224),      # 中心裁剪到224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_dataset = own_dataset(root="D:/testPython/暑期课程/pet/images", preprocess=transform_train)#实例化

# 定义划分比例（如 80% 训练，20% 测试）
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 随机拆分
trainset, testset = random_split(full_dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

# 可对测试集单独应用不同的预处理（需重新包装数据集）
testset.dataset.preprocess = transform_test  # 修改测试集的预处理

img,label=trainset[0]
# plt.imshow(img)
# print(img.shape)
import numpy as np
image_transposed = np.transpose(img, (1, 2, 0))  # 新形状 (32, 32, 3)
plt.imshow(image_transposed)
plt.show()

# Model
print('==> Building model..')
# net = ResNet18()

# 1. 加载预训练 ResNet18 模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 使用官方预训练权重

# 2. 冻结所有参数（除了最后一层）
for param in model.parameters():
    param.requires_grad = False  # 冻结所有层

# 3. 替换最后一层（全连接层）
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 37)  # pet有 37 个类别
model.fc.requires_grad = True  # 确保最后一层可训练

# model = model.to('cpu')
# 4. 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)  # loss=L+\lambda||w||^2
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 5. 定义损失函数和优化器（仅优化最后一层）
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.fc.parameters(), lr=0.001, momentum=0.9)  # 只传递最后一层的参数
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # for param in net.parameters():
        #     print(param.data,param.grad)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # tqdm
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    avg_loss = train_loss / len(trainloader)
    avg_acc = 100. * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(avg_acc)

def test(epoch):
    global best_acc
    model.eval()
    # for param in net.parameters():
    #     param.requires_grad = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    avg_loss = test_loss / len(testloader)
    avg_acc = 100. * correct / total
    test_losses.append(avg_loss)
    test_accuracies.append(avg_acc)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch + 10):
    train(epoch)
    test(epoch)
    scheduler.step()

plt.figure(figsize=(12, 4))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')  # 保存图像
plt.show()  # 显示图像
# # 7. 训练循环（仅训练最后一层）
# model.train()  # 启用训练模式
# for epoch in range(5):  # 示例：训练 5 个 epoch
#     for inputs, labels in trainloader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#     print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
