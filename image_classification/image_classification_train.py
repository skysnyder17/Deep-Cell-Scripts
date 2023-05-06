import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from deepcell import datasets
from deepcell import model_zoo
from deepcell.utils import plot_history

# Load the training and testing datasets
(X_train, y_train), (X_test, y_test) = datasets.imagenet.load_data()

# Normalize the pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define the input shape of the model
input_shape = (224, 224, 3)

# Load the pre-trained image classification model
model_name = 'resnet50'
num_classes = 1000
model = model_zoo.ClassificationModel(
    backbone=model_name,
    input_shape=input_shape,
    num_classes=num_classes,
    weights='imagenet'
)

# Freeze the backbone layers
for layer in model.backbone.layers:
    layer.trainable = False

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Plot the training history
plot_history(history)

# Save the model
model_dir = 'path/to/save/model'
model_path = os.path.join(model_dir, 'image_classification_model.h5')
model.save(model_path)
