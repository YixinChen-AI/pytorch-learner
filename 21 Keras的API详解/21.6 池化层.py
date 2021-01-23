import tensorflow as tf
x = tf.random.normal((4,28,28,3))
y = tf.keras.layers.MaxPooling2D(
    pool_size=(2,2),
    strides = 1,
    padding='same')
print(y(x).shape)
#-------------------------------------
import tensorflow as tf
x = tf.random.normal((4,28,28,3))
y = tf.keras.layers.GlobalMaxPooling2D()
print(y(x).shape)