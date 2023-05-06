import numpy as np
import skimage.data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from deepcell import denoise, deconvolution

# Load a training image
image = skimage.data.camera()

# Add Gaussian noise to the image
noisy_image = image + 0.1 * np.random.randn(*image.shape)

# Normalize the image intensities to [0, 1]
image = image / 255.0
noisy_image = np.clip(noisy_image, 0, 255) / 255.0

# Define the denoising model architecture
denoise_model = keras.Sequential([
    keras.layers.Input(shape=image.shape),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')
])

# Compile the denoising model
denoise_model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the denoising model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
denoise_model.fit(
    x=np.expand_dims(noisy_image, axis=0),
    y=np.expand_dims(image, axis=0),
    batch_size=1,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Define the deconvolution model architecture
deconvolution_model = keras.Sequential([
    keras.layers.Input(shape=image.shape),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')
])

# Compile the deconvolution model
deconvolution_model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the deconvolution model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
deconvolution_model.fit(
    x=np.expand_dims(denoise.denoise_image(noisy_image, denoise_model), axis=0),
    y=np.expand_dims(image, axis=0),
    batch_size=1,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Save the trained models
denoise_model.save('denoise_model.h5')
deconvolution_model.save('deconvolution_model.h5')
