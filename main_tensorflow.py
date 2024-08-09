import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import pandas as pd
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# 定义情绪类别及其对应目录
emotion_dirs = {
    'angry': './train/angry/',
    'disgusted': './train/disgusted/',
    'fearful': './train/fearful/',
    'happy': './train/happy/',
    'neutral': './train/neutral/',
    'sad': './train/sad/',
    'surprised': './train/surprised/'
}

# 遍历字典并打印每个类别前10个文件
for emotion, dir_path in emotion_dirs.items():
    num_images = len(os.listdir(dir_path))
    print(f'total training {emotion} images:', num_images)
    files = os.listdir(dir_path)
    print(f'{emotion} files:', files[:5])
    img = cv2.imread(os.path.join(dir_path, files[0]))
    plt.figure(figsize=(2, 2))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 确保图像是RGB格式
    plt.axis('Off')
    plt.title(dir_path.split('/')[2])
    plt.show()

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

TRAINING_DIR = './train/'
training_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 值将在执行其他处理前乘到整个图像上
    rotation_range=40,  # 整数，数据提升时图片随机转动的角度
    width_shift_range=0.2,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
    height_shift_range=0.2,  # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度
    shear_range=0.2,  # 浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度
    zoom_range=0.2,  # 用来进行随机的放大
    validation_split=0.25,
    horizontal_flip=True,  # 布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
    fill_mode='nearest'  # 'constant','nearest','reflect','wrap'之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
)
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    subset='training',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    subset='validation',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

class_indices = train_generator.class_indices
print(class_indices)

# 计算每个类别的权重
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),  # 卷积
    tf.keras.layers.MaxPooling2D(2, 2),  # 池化

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积
    tf.keras.layers.MaxPooling2D(2, 2),  # 池化

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # 卷积
    tf.keras.layers.MaxPooling2D(2, 2),  # 池化

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),  # 全连接
    tf.keras.layers.Dropout(0.5),  # 一种防止神经网络过拟合的手段
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    class_weight=class_weights,  # 传递class_weights
    verbose=1
)

model.save('./model/smallmodel.keras')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.savefig("./result/train_loss.png")
plt.show()

model = load_model("./model/smallmodel.keras")

# Predictions and Confusion Matrix
validation_generator.reset()
predictions = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

conf_matrix = confusion_matrix(true_classes, predicted_classes)

# 打印分类报告
print('Classification Report')
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=1))

# 绘制混淆矩阵
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
plt.savefig("./result/matrix.png")