import tensorflow as tf
import tensorflow as tf

class MyInitializer(tf.keras.initializers.Initializer):

    def __init__(self, mean, stddev):
      self.mean = mean
      self.stddev = stddev

    def __call__(self, shape, dtype=None):
      return tf.random.normal(
          shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

    def get_config(self):  # To support serialization
      return {'mean': self.mean, 'stddev': self.stddev}

input_shape = (4, 28, 28, 3)
initializer = MyInitializer(mean=0., stddev=1.)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
    filters=2,kernel_size=3,
    activation='selu',padding='same',
    kernel_initializer=initializer,
    bias_initializer=initializer
)
print(y(x).shape)