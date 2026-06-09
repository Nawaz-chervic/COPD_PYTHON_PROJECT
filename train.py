# =====================================
# COPD Detection - Improved Training
# Handles imbalanced dataset
# =====================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# =====================================
# Check Dataset
# =====================================

DATA_DIR = 'dataset' if os.path.exists('dataset') else 'Dataset' if os.path.exists('Dataset') else None
if DATA_DIR is None:
    print("ERROR: Dataset folder not found! Create 'dataset/' or 'Dataset/' with class subfolders.")
    exit()

print("\nDataset contents:")
class_counts = {}
for cls in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, cls)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[cls] = count
        print(f"  {cls}: {count} images")

# =====================================
# Compute Class Weights (fixes imbalance)
# =====================================

total = sum(class_counts.values())
n_classes = len(class_counts)
class_names = sorted(class_counts.keys())   # alphabetical = how Keras reads them

class_weights = {}
for i, cls in enumerate(class_names):
    weight = total / (n_classes * class_counts[cls])
    class_weights[i] = weight
    print(f"  Class weight [{i}] {cls}: {weight:.4f}")

print(f"\nClass weights: {class_weights}")

# =====================================
# Data Augmentation
# =====================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"\nClass indices: {train_data.class_indices}")
print(f"Training samples:   {train_data.samples}")
print(f"Validation samples: {validation_data.samples}")

# =====================================
# Remap class weights to match
# Keras class_indices order
# =====================================

class_indices = train_data.class_indices  # e.g. {'COPD': 0, 'Normal': 1}
remapped_weights = {}
for cls_name, idx in class_indices.items():
    remapped_weights[idx] = total / (n_classes * class_counts[cls_name])

print(f"Remapped class weights: {remapped_weights}")

# =====================================
# Transfer Learning — MobileNetV2
# =====================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================
# Callbacks
# =====================================

checkpoint = ModelCheckpoint(
    'copd_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# =====================================
# Phase 1: Train top layers only
# =====================================

print("\n--- Phase 1: Training top layers (15 epochs) ---")

history1 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    class_weight=remapped_weights,      # ← fixes imbalance
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# =====================================
# Phase 2: Fine-tune last 30 layers
# =====================================

print("\n--- Phase 2: Fine-tuning last 30 layers (15 epochs) ---")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    class_weight=remapped_weights,      # ← fixes imbalance
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# =====================================
# Save Final Model
# =====================================

model.save('copd_model.h5')
print("\n✅ Model saved as copd_model.h5")

# =====================================
# Evaluate on validation set
# =====================================

print("\n--- Final Evaluation ---")
loss, accuracy = model.evaluate(validation_data)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss:     {loss:.4f}")

# =====================================
# Plot Results
# =====================================

acc     = history1.history['accuracy']     + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_h  = history1.history['loss']         + history2.history['loss']
val_loss_h = history1.history['val_loss']  + history2.history['val_loss']

split = len(history1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(acc, label='Train Accuracy')
axes[0].plot(val_acc, label='Val Accuracy')
axes[0].axvline(x=split, color='gray', linestyle='--', label='Fine-tune start')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(loss_h, label='Train Loss')
axes[1].plot(val_loss_h, label='Val Loss')
axes[1].axvline(x=split, color='gray', linestyle='--', label='Fine-tune start')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
print("Graph saved as training_results.png")