# ======================================
# Step 1: Import required libraries
# ======================================
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# ======================================
# Step 2: Create output folder
# ======================================
output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

# ======================================
# Step 3: Load and preprocess CIFAR-10 dataset
# ======================================
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# ======================================
# Step 4: Build the CNN model
# ======================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# ======================================
# Step 5: Compile the model
# ======================================
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ======================================
# Step 6: Train the model
# ======================================
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

# ======================================
# Step 7: Evaluate model on test data
# ======================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# Save test accuracy to a text file
with open(os.path.join(output_folder, "accuracy.txt"), "w") as f:
    f.write(f"Test accuracy: {test_acc:.4f}\n")

# ======================================
# Step 8: Plot and save training graph
# ======================================
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "accuracy_plot.png"))
plt.show()

# ======================================
# Step 9: Predict and save probabilities
# ======================================
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(x_test)

# Save predictions for first 10 samples
np.save(os.path.join(output_folder, "predictions.npy"), predictions[:10])
