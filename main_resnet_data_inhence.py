import os
import random
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50

# 设置随机种子以确保结果的可重复性
torch.manual_seed(2024)
warnings.filterwarnings("ignore")


# 定义函数：统计数据集中各类别的样本数量
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


# 数据增强和预处理
# 包含：图像调整为224x224、灰度图转换为三通道、随机水平翻转、随机旋转、颜色抖动、归一化
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 验证集的预处理（不包含数据增强，仅执行标准化）
validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载数据集并应用数据增强
full_dataset = datasets.ImageFolder(root='./train/', transform=train_transform)

# 统计各类别的样本数量并获取类别到索引的映射
class_counts, class_to_idx = calculate_class_distribution(full_dataset)

# 计算每个类别的样本权重，用于处理数据不平衡问题
class_weights = [1.0 / class_counts[class_name] for class_name in full_dataset.classes]

# 为整个数据集创建样本权重
full_sample_weights = [class_weights[label] for _, label in full_dataset]

# 将数据集划分为训练集和验证集（80%训练集，20%验证集）
train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size
train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

# 根据训练集的索引创建样本权重
train_sampler_weights = [full_sample_weights[idx] for idx in train_dataset.indices]

# 使用 WeightedRandomSampler 进行加权采样，确保训练时的采样平衡
train_sampler = WeightedRandomSampler(train_sampler_weights, num_samples=len(train_sampler_weights), replacement=True)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)


# 定义情感分类器模型，基于ResNet50
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        self.resnet = resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


# 混合精度训练设置
model = EmotionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()

# 检查是否可以使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练参数设置
epochs = 100
best_accuracy = 0.0

# 模型训练与验证循环
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

    accuracy = 100 * correct / total
    print(f'第 {epoch + 1} 轮, 损失: {running_loss / len(train_loader)}, 验证准确率: {accuracy}%')

    # 保存验证集上效果最好的模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), './model/emotion_classifier_resnet_best.pth')

print(f'最佳验证准确率: {best_accuracy}%')

# 加载最优模型以进行推理
model = EmotionClassifier()
model.load_state_dict(torch.load('./model/emotion_classifier_resnet_best.pth'))
model.to(device)
model.eval()

# 准备预测，计算混淆矩阵和分类报告
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)

# 打印分类报告
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


# 测试和预测单张图像
def predict_image(image_path):
    img = Image.open(image_path)
    img = validation_transform(img).float()
    img = img.unsqueeze(0)
    output = model(img.to(device))
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


# 使用示例
emotion = predict_image('./test/00001.png')
print(f'预测情感类别索引: {emotion}')  # 将索引映射到对应的情感类别名称


# 预测并保存结果的函数
def predict_and_save_results(test_dir, output_csv):
    results = []  # 用于存储所有预测结果的列表

    # 获取测试集目录中的所有图片文件
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for file_name in test_images:
        image_path = os.path.join(test_dir, file_name)
        # 调用预测函数获取预测结果
        predicted_num = predict_image(image_path)

        # 根据预测结果的索引映射到对应的情感类别名称
        if predicted_num == 0:
            face = 'angry'
        elif predicted_num == 1:
            face = 'disgusted'
        elif predicted_num == 2:
            face = 'fearful'
        elif predicted_num == 3:
            face = 'happy'
        elif predicted_num == 4:
            face = 'neutral'
        elif predicted_num == 5:
            face = 'sad'
        elif predicted_num == 6:
            face = 'surprised'

        # 将结果添加到结果列表中
        results.append({'name': file_name, 'label': face})

    # 将结果列表转换为DataFrame
    dataframe = pd.DataFrame(results)

    # 保存DataFrame为CSV文件
    dataframe.to_csv(output_csv, index=False)
    print(f"预测结果已保存到 {output_csv}")


# 定义测试集路径和输出CSV文件路径
test_dir = './test'
output_csv = './result/emotion_predictions.csv'

# 运行预测并保存结果
predict_and_save_results(test_dir, output_csv)
