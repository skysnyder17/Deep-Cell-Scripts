import tensorflow as tf
from deepcell.applications import SuperResolution

# Load the super-resolution model
model = SuperResolution()

# Load a low-resolution test image
low_res_img = tf.keras.preprocessing.image.load_img("low_res_image.png", target_size=(256, 256))
low_res_img = tf.keras.preprocessing.image.img_to_array(low_res_img)
low_res_img = low_res_img.astype('float32') / 255.0

# Upscale the image using the super-resolution model
high_res_img = model.predict(low_res_img)

# Save the high-resolution image
tf.keras.preprocessing.image.save_img("high_res_image.png", high_res_img[0])
