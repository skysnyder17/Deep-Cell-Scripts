import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from deepcell import model_zoo

# Load the saved model
model_dir = 'path/to/saved/model'
model_path = os.path.join(model_dir, 'image_classification_model.h5')
model = keras.models.load_model(model_path)

# Load an example image for testing
image_path = 'path/to/example/image'
img = keras.preprocessing.image.load_img(
    image_path,
    target_size=(224, 224)
)
img_arr = keras.preprocessing.image.img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr /= 255.0

# Make a prediction on the example image
pred = model.predict(img_arr)[0]
predicted_class = np.argmax(pred)
class_names = model_zoo.imagenet.get_imagenet_label_dict()
predicted_label = class_names[predicted_class]

# Plot the example image and the predicted label
plt.imshow(img)
plt.axis('off')
plt.title(predicted_label)
plt.show()
