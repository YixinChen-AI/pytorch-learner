import tensorflow as tf
import numpy as np
x = tf.constant(np.arange(10).reshape(5,2)*10,
                dtype=tf.float32)
print(x)
y = tf.keras.layers.LayerNormalization(axis=1)
print(y(x))