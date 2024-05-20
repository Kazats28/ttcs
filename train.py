import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Đường dẫn tới dữ liệu
train_dir = './train'
validation_dir = './validation'
test_dir = './test'

# Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Thay đổi kích thước đầu vào
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),  # Thay đổi kích thước đầu vào
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),  # Thay đổi kích thước đầu vào
    batch_size=32,
    class_mode='binary'
)

# Thiết kế mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.4),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.4),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.4),
    Conv2D(512, (1, 1), activation='relu'),
    Flatten(),
    Dropout(0.4),
    Dense(120, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình với Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Đánh giá mô hình trên tập kiểm định
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Dự đoán trên tập kiểm tra
test_generator.reset()
Y_pred = model.predict(test_generator, len(test_generator))
y_pred = (Y_pred > 0.5).astype(int)

# Lấy nhãn thực tế
y_true = test_generator.classes

# Tính toán các chỉ số
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Classification Report')
target_names = ['cat', 'dog']
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=target_names))

# Lưu mô hình
model.save('cat_dog_classifier.h5')
