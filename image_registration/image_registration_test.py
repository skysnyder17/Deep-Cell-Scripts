import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from deepcell import registration

# Load the test data
fixed_image = np.load('path/to/fixed/image.npy')
moving_image = np.load('path/to/moving/image.npy')

# Normalize the image intensities to [0, 1]
fixed_image /= 255.0
moving_image /= 255.0

# Load the trained model
model = keras.models.load_model('path/to/saved/model', compile=False)

# Set the registration parameters
params = registration.Params()
params.registration_algorithm = registration.RegistrationAlgorithm.FFT

# Perform the registration
transform = registration.register(fixed_image, moving_image, model, params)

# Apply the transformation to the moving image
registered_image = registration.apply_transform(moving_image, transform)

# Visualize the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(fixed_image, cmap='gray')
axs[0].set_title('Fixed Image')
axs[1].imshow(moving_image, cmap='gray')
axs[1].set_title('Moving Image')
axs[2].imshow(registered_image, cmap='gray')
axs[2].set_title('Registered Image')
plt.show()
