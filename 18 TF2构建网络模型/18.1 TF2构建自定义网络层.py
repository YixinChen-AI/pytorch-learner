import tensorflow as tf
import tensorflow.keras as keras

class MyLayer(keras.layers.Layer):
    def __init__(self, input_dim=32, output_dim=32):
        super(MyLayer, self).__init__()

        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype=tf.float32),
            trainable=True) # 如果是false则是不参与梯度下降的变量

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(output_dim), dtype=tf.float32), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


x = tf.ones((3,5))
my_layer = MyLayer(input_dim=5,
                   output_dim=10)
out = my_layer(x)
print(out.shape)