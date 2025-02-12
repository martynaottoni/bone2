# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:53:27 2025

@author: Martyna Ottoni
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os

train_dir = r"C:\Users\Ottoni\Desktop\bones\Martyna training\bone2\Train"
test_dir = r"C:\Users\Ottoni\Desktop\bones\Martyna training\bone2\Test"

def create_generators(train_dir, test_dir, batch_size_parameter):
    train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    ).flow_from_directory(
        directory=train_dir,
        target_size=(224, 224),
        batch_size=batch_size_parameter,
        class_mode='categorical',
        shuffle=True
    )

    test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=test_dir,
        target_size=(224, 224),
        batch_size=batch_size_parameter,
        class_mode='categorical',
        shuffle=False
    )
    return train_data, test_data

batch_size = 32
# Create data generators
train_data, test_data = create_generators(train_dir, test_dir, batch_size)

# Retrieve class information (important for output_signature)
class_indices = train_data.class_indices
class_names = list(class_indices.keys())
num_classes = len(class_names)

# Convert generators to tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
).cache()

# Build the model
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'), # Another Dense layer with 128 neurons
    Dropout(0.1),
    Dense(num_classes, activation='softmax')
])

# Compile the model and define callbacks
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

img_number = 989
test_img_number = 140

# Compute steps_per_epoch
steps_per_epoch = img_number // batch_size
if img_number % batch_size != 0:
    steps_per_epoch += 1 # Round up

# Compute validation_steps
validation_steps = test_img_number // batch_size
if test_img_number % batch_size != 0:
    validation_steps += 1 # Round up

# Train the model using datasets
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=100,
    steps_per_epoch=steps_per_epoch, # Limit number of steps per epoch
    validation_steps=validation_steps, # Limit number of validation steps
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Visualize training results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate the model
val_loss, val_accuracy = model.evaluate(test_data)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
