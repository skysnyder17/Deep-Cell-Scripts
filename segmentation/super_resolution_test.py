import os
import numpy as np

from main_script import segment_tissue_dataset

# Set up the data directory
data_dir = 'path/to/dataset'

# Call the main script to segment the dataset
segment_tissue_dataset(data_dir)

# Load the segmentation model
model_path = 'segmentation_model.h5'
model = keras.models.load_model(model_path)

# Load the test data
test_dir = os.path.join(data_dir, 'test')
X_test, y_test = utils.load_test_data(test_dir)

# Make predictions with the model
y_pred = model.predict(X_test)

# Compute the evaluation metrics
metrics = model.evaluate(X_test, y_test)

# Print the evaluation metrics
print('Evaluation Metrics:')
for i, metric_name in enumerate(model.metrics_names):
    print('{}: {}'.format(metric_name, metrics[i]))

# Visualize the results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(X_test[0])
ax[0].set_title('Input Image')
ax[1].imshow(y_test[0, ..., 0], cmap='gray')
ax[1].set_title('Ground Truth')
ax[2].imshow(y_pred[0, ..., 0], cmap='gray')
ax[2].set_title('Predicted Segmentation')
plt.show()
