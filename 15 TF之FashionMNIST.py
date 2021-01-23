import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('train_images shape:',train_images.shape)
print('train_labels shape:',train_labels.shape)
print('test_images shape:',test_images.shape)
print('test_labels shape:',test_labels.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

# 模型搭建
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# %% [code]
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

# %% [code]
predictions = model.predict(test_images)
predictions[0]

# %% [code]
