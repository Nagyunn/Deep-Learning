import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ================== 1. CẤU HÌNH GPU AMD (DIRECTML) ==================
tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2600)]
        )
        print("✅ Đã kích hoạt GPU AMD RX 5500M.")
    except RuntimeError as e:
        print(f"❌ Lỗi cấu hình GPU: {e}")

# ================== 2. CẤU HÌNH ĐƯỜNG DẪN ==================
BASE_DIR = r"C:\Users\ADMIN\Documents\flower_data"
train_dir = os.path.join(BASE_DIR, "train")
valid_dir = os.path.join(BASE_DIR, "val")

BATCH_SIZE = 32
IMG_SIZE = (227, 227)

# ================== 3. DATA AUGMENTATION (GIỮ NGUYÊN) ==================
train_augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,
    fill_mode='nearest'
)

valid_rescaler = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = train_augmentor.flow_from_directory(
    train_dir, batch_size=BATCH_SIZE, target_size=IMG_SIZE, class_mode='categorical', shuffle=True
)

valid_data_gen = valid_rescaler.flow_from_directory(
    valid_dir, batch_size=BATCH_SIZE, target_size=IMG_SIZE, class_mode='categorical', shuffle=False
)

num_classes = train_data_gen.num_classes
class_names = list(train_data_gen.class_indices.keys())

# ================== 4. MODEL ALEXNET  ==================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(227, 227, 3)),
    tf.keras.layers.Conv2D(96, 11, strides=4, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Conv2D(256, 5, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# ================== 5. COMPILE & CALLBACKS ==================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback lưu bản tốt nhất (Dùng .keras cho ổn định khi train)
checkpoint = ModelCheckpoint(
    os.path.join(BASE_DIR, "alexnet_best_weights.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3, patience=3,
    min_lr=1e-6
)

# ================== 6. HUẤN LUYỆN ==================
history = model.fit(
    train_data_gen,
    epochs=70,
    validation_data=valid_data_gen,
    callbacks=[early_stop, lr_reduce, checkpoint]
)

# ================== 7. LƯU SONG SONG HAI ĐỊNH DẠNG ==================
model.save(os.path.join(BASE_DIR, "alexnet_flower_final.keras"))

np.save(os.path.join(BASE_DIR, "class_names.npy"), class_names)

print("\n🎉 Đã lưu mô hình dưới 2 định dạng: .keras")
print(f"📍 Vị trí: {BASE_DIR}")

# ================== 8. BIỂU ĐỒ ==================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend()
plt.show()