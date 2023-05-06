import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from deepcell import registration

# Load the two images to be registered
fixed_image_path = 'path/to/fixed/image'
moving_image_path = 'path/to/moving/image'

fixed_image = keras.preprocessing.image.load_img(fixed_image_path, grayscale=True)
moving_image = keras.preprocessing.image.load_img(moving_image_path, grayscale=True)

fixed_image_arr = keras.preprocessing.image.img_to_array(fixed_image)
moving_image_arr = keras.preprocessing.image.img_to_array(moving_image)

# Normalize the image intensities to [0, 1]
fixed_image_arr /= 255.0
moving_image_arr /= 255.0

# Set the registration parameters
params = registration.Params()
params.registration_algorithm = registration.RegistrationAlgorithm.FFT

# Compute the transformation between the two images
transformation = registration.compute_registration(
    fixed_image_arr, moving_image_arr, params=params)

# Apply the transformation to the moving image
registered_image = registration.apply_transformation(
    moving_image_arr, transformation)

# Plot the fixed and registered images side-by-side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(fixed_image_arr[:, :, 0], cmap='gray')
ax[0].set_title('Fixed image')
ax[1].imshow(registered_image[:, :, 0], cmap='gray')
ax[1].set_title('Registered image')
plt.show()
