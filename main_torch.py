import os
import random
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast


# 设置随机种子以确保结果的可重复性
torch.manual_seed(2024)
warnings.filterwarnings("ignore")


# 统计数据集中各类别的样本数量
def calculate_class_distribution(dataset):
    class_counts = defaultdict(int)

    for _, label in dataset:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1

    class_to_idx = dataset.class_to_idx
    print("类别到索引的映射:", class_to_idx)

    for class_name, count in class_counts.items():
        print(f"类别: {class_name}, 数量: {count}")

    return class_counts, class_to_idx


# 数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

validation_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集并进行统计分析
full_dataset = datasets.ImageFolder(root='./train/', transform=transform)
class_counts, class_to_idx = calculate_class_distribution(full_dataset)

# 数据集划分为训练集和验证集（80% 训练集，20% 验证集）
train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size
train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)


# 定义情感分类模型
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 卷积层1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 卷积层2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 卷积层3
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # 全连接层1
        self.fc2 = nn.Linear(256, 7)  # 输出层
        self.dropout = nn.Dropout(0.5)  # Dropout 层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 模型训练部分
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()

epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'第 {epoch + 1} 轮, 损失: {running_loss / len(train_loader):.4f}, '
          f'验证准确率: {100 * correct / total:.2f}%')

# 保存训练好的模型
torch.save(model.state_dict(), './model/emotion_classifier.pth')


# 验证模型表现
model = EmotionClassifier()
model.load_state_dict(torch.load('./model/emotion_classifier.pth'))
model.to(device)
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 计算并显示混淆矩阵和分类报告
conf_matrix = confusion_matrix(all_labels, all_preds)
print('分类报告')
class_names = list(class_to_idx.keys())
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=1))

# 绘制混淆矩阵
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# 图像预测函数
def predict_image(image_path):
    img = Image.open(image_path)
    img = validation_transform(img).float()
    img = img.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()


# 示例使用：预测单张图像的情感类别
emotion = predict_image('./test/00001.png')
print(f'预测情感类别索引: {emotion}')  # 根据索引映射至对应的情感类别名称
