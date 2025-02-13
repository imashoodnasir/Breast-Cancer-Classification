import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define directories
train_dir = "dataset/train"
test_dir = "dataset/test"
img_size = (224, 224)
batch_size = 32
num_classes = 2  # Adjust according to dataset

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.5)  # 50% split for train/test

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset="training"
)

test_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation"
)

# Build the model
model = build_32_layer_cnn(input_shape=(224, 224, 3), num_classes=num_classes)

# Define optimizer with Momentum and L2 regularization
optimizer = SGD(learning_rate=0.001, momentum=0.7, nesterov=True)

# Compile the model with loss functions
model.compile(
    optimizer=optimizer,
    loss=[CategoricalCrossentropy(), MeanSquaredError()],
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=150,
    batch_size=batch_size
)

# Save model
model.save("breast_cancer_cnn.h5")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
