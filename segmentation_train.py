import os
import numpy as np

from deepcell import datasets
from deepcell import model_zoo
from deepcell import metrics
from deepcell import losses
from deepcell import utils

# Set up the data directories
data_dir = 'path/to/dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Set up the data generators
train_data = utils.get_data_generator(
    train_dir,
    rotation_range=180,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    dim_ordering='tf'
)
val_data = utils.get_data_generator(
    val_dir,
    dim_ordering='tf'
)

# Define the model architecture
model = model_zoo.bn_feature_net_61(
    input_shape=train_data.shape[1:],
    norm_method='whole_image',
    num_classes=len(np.unique(train_data.y))
)

# Compile the model
model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer='adam',
    metrics=[
        metrics.BinaryAccuracy(name='acc'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall')
    ]
)

# Train the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_acc',
            save_best_only=True,
            save_weights_only=True
        )
    ]
)

# Save the model
model.save('segmentation_model.h5')
