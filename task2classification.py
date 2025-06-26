# ============================ 
# Step 1: Import required libraries
# ============================

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Access keras modules using tf.keras
datasets = tf.keras.datasets
layers = tf.keras.layers
models = tf.keras.models

# ============================
# Step 2: Create output folder
# ============================

output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

# ============================
# Step 3: Load and preprocess CIFAR-10 dataset
# ============================

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# ============================
# Step 4: Build CNN model
# ============================

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 output classes
])

# ============================
# Step 5: Compile the model
# ============================

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ============================
# Step 6: Train the model
# ============================

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# ============================
# Step 7: Evaluate the model
# ============================

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# ============================
# Step 8: Save accuracy plot
# ============================

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.grid(True)

# Save the plot
plot_path = os.path.join(output_folder, 'accuracy_plot.png')
plt.savefig(plot_path)
print(f"📊 Accuracy plot saved at: {plot_path}")
