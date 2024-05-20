import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Đường dẫn tới dữ liệu
train_dir = './train'
validation_dir = './validation'
test_dir = './test'

# Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Kích thước đầu vào của mô hình đã huấn luyện
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),  # Kích thước đầu vào của mô hình đã huấn luyện
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),  # Kích thước đầu vào của mô hình đã huấn luyện
    batch_size=32,
    class_mode='binary'
)

# Tải lại mô hình từ file .h5
model = load_model('cat_dog_classifier.h5')

# Đánh giá mô hình trên tập huấn luyện
train_loss, train_acc = model.evaluate(train_generator)
print(f'Train Accuracy: {train_acc * 100:.2f}%')

# Đánh giá mô hình trên tập validation
validation_loss, validation_acc = model.evaluate(validation_generator)
print(f'Validation Accuracy: {validation_acc * 100:.2f}%')

# Đánh giá mô hình trên tập kiểm định
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Dự đoán trên tập kiểm tra
test_generator.reset()
Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int)

# Lấy nhãn thực tế
y_true = test_generator.classes

# Tính toán các chỉ số
print('Test Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Test Classification Report')
target_names = ['cat', 'dog']  # Khớp với nhãn ảnh của bạn
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=target_names))

# Tạo DataFrame từ report dictionary
df_report_test = pd.DataFrame(report).transpose()

# Tạo DataFrame chứa bảng kết quả cho tập kiểm định
df_results_test = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Cat)', 'Precision (Dog)', 'Recall (Cat)', 'Recall (Dog)', 'F1-Score (Cat)', 'F1-Score (Dog)'],
    'Test': [df_report_test.loc['accuracy'], df_report_test.loc['cat', 'precision'], df_report_test.loc['dog', 'precision'], df_report_test.loc['cat', 'recall'], df_report_test.loc['dog', 'recall'], df_report_test.loc['cat', 'f1-score'], df_report_test.loc['dog', 'f1-score']]
})

# Chuyển đổi các giá trị thành phần trăm
df_results_test['Test'] = df_results_test['Test'] * 100

# Lưu DataFrame chứa bảng kết quả cho tập kiểm định
df_results_test.to_csv('model_evaluation_results_test.csv', index=False)

# Dự đoán trên tập validation
validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = (Y_pred > 0.5).astype(int)

# Lấy nhãn thực tế
y_true = validation_generator.classes

# Tính toán các chỉ số
print('Validation Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Validation Classification Report')
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=target_names))

# Tạo DataFrame từ report dictionary
df_report_validation = pd.DataFrame(report).transpose()

# Tạo DataFrame chứa bảng kết quả cho tập validation
df_results_validation = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Cat)', 'Precision (Dog)', 'Recall (Cat)', 'Recall (Dog)', 'F1-Score (Cat)', 'F1-Score (Dog)'],
    'Validation': [df_report_validation.loc['accuracy'], df_report_validation.loc['cat', 'precision'], df_report_validation.loc['dog', 'precision'], df_report_validation.loc['cat', 'recall'], df_report_validation.loc['dog', 'recall'], df_report_validation.loc['cat', 'f1-score'], df_report_validation.loc['dog', 'f1-score']]
})

# Chuyển đổi các giá trị thành phần trăm
df_results_validation['Validation'] = df_results_validation['Validation'] * 100

# Lưu DataFrame chứa bảng kết quả cho tập validation
df_results_validation.to_csv('model_evaluation_results_validation.csv', index=False)

# Dự đoán trên tập huấn luyện
train_generator.reset()
Y_pred = model.predict(train_generator)
y_pred = (Y_pred > 0.5).astype(int)

# Lấy nhãn thực tế
y_true = train_generator.classes

# Tính toán các chỉ số trên tập huấn luyện
print('Train Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Train Classification Report')
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=target_names))

# Tạo DataFrame từ report dictionary
df_report_train = pd.DataFrame(report).transpose()

# Tạo DataFrame chứa bảng kết quả cho tập huấn luyện
df_results_train = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Cat)', 'Precision (Dog)', 'Recall (Cat)', 'Recall (Dog)', 'F1-Score (Cat)', 'F1-Score (Dog)'],
    'Train': [df_report_train.loc['accuracy'], df_report_train.loc['cat', 'precision'], df_report_train.loc['dog', 'precision'], df_report_train.loc['cat', 'recall'], df_report_train.loc['dog', 'recall'], df_report_train.loc['cat', 'f1-score'], df_report_train.loc['dog', 'f1-score']]
})

# Chuyển đổi các giá trị thành phần trăm
df_results_train['Train'] = df_results_train['Train'] * 100

# Lưu DataFrame chứa bảng kết quả cho tập huấn luyện
df_results_train.to_csv('model_evaluation_results_train.csv', index=False)

# Kết hợp các bảng kết quả thành một bảng duy nhất
df_results_combined = pd.merge(df_results_train, df_results_validation, on='Metric')
df_results_combined = pd.merge(df_results_combined, df_results_test, on='Metric')

# In bảng kết quả
print(df_results_combined)

# Lưu DataFrame vào tập tin CSV
df_results_combined.to_csv('model_evaluation_results_combined.csv', index=False)
