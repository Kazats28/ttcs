import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Tải mô hình đã lưu
model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Class indices mapping
class_indices = {0: 'cat', 1: 'dog'}

def load_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path)
    img_resized = img.resize((64, 64))  # Kích thước ảnh đầu vào của mô hình
    img_array = np.array(img_resized) / 255.0  # Tiền xử lý ảnh
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = 'dog' if prediction > 0.5 else 'cat'

    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    result_label.config(text=f'Predicted Class: {predicted_class}')

# Tạo cửa sổ giao diện
root = tk.Tk()
root.title('Cat and Dog')

# Tạo các widget
upload_btn = tk.Button(root, text='Select Image', command=load_image)
upload_btn.pack()

image_label = Label(root)
image_label.pack()

result_label = Label(root, text='', font=('Arial', 20))
result_label.pack()

# Chạy ứng dụng
root.mainloop()
