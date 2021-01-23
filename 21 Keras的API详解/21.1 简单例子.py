import tensorflow as tf
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
    filters=2,kernel_size=3,
    activation='relu',padding='same'
)
print(y(x).shape)