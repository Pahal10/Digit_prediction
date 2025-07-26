# Make sure to install these first:
# pip install tensorflow idx2numpy

import idx2numpy
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

base_path = #replace this with the archive folder's path

# Load IDX files
X_train = idx2numpy.convert_from_file(base_path + "\\train-images-idx3-ubyte\\train-images-idx3-ubyte")
y_train = idx2numpy.convert_from_file(base_path + "\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte")
X_test = idx2numpy.convert_from_file(base_path + "\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte")
y_test = idx2numpy.convert_from_file(base_path + "\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte")

# Reshape and normalize
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

#accuracy can be changed by addition/deletion of layers
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5)
]

# To check total Training time
start = time.time()
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, callbacks=callbacks)
end = time.time()

minutes, seconds = divmod(end - start, 60)
print(f"\n✅ Training Time: {int(minutes)} min {int(seconds)} sec")

loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}")
