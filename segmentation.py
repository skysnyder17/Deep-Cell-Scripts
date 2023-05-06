import os
import numpy as np
import matplotlib.pyplot as plt

from deepcell import datasets
from deepcell import model_zoo
from deepcell import metrics
from deepcell import layers
from deepcell import losses
from deepcell import utils

# Load the data
data_dir = 'path/to/dataset'
(X_train, y_train), (X_test, y_test) = datasets.imagenet.load_data(data_dir)

# Preprocess the data
X_train, y_train = utils.format_data(X_train, y_train)
X_test, y_test = utils.format_data(X_test, y_test)

# Define the model architecture
model = model_zoo.bn_feature_net_61(
    input_shape=X_train.shape[1:],
    norm_method='whole_image',
    num_classes=len(np.unique(y_train))
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
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    callbacks=[
        callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_acc',
            save_best_only=True,
            save_weights_only=True
        )
    ]
)

# Make predictions with the trained model
y_pred = model.predict(X_test)

# Visualize the results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(X_test[0])
ax[0].set_title('Input Image')
ax[1].imshow(y_test[0, ..., 0], cmap='gray')
ax[1].set_title('Ground Truth')
ax[2].imshow(y_pred[0, ..., 0], cmap='gray')
ax[2].set_title('Predicted Segmentation')
plt.show()

# Save the model
model.save('segmentation_model.h5')
