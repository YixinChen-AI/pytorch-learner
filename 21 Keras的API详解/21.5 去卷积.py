import tensorflow as tf
from tensorflow import keras

input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = keras.layers.Conv2DTranspose(
    filters=10,kernel_size=3,strides=1,padding='same')
print(y(x).shape)

input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = keras.layers.Conv2DTranspose(
    filters=10,kernel_size=3,strides=1)
print(y(x).shape)

input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = keras.layers.Conv2DTranspose(
    filters=10,kernel_size=3,strides=2)
print(y(x).shape)

input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = keras.layers.Conv2DTranspose(
    filters=10,kernel_size=3,strides=2,padding='same')
print(y(x).shape)

