import tensorflow as tf
import tensorflow.keras as keras

class CBR(keras.layers.Layer):
    def __init__(self,output_dim):
        super(CBR,self).__init__()
        self.conv = keras.layers.Conv2D(filters=output_dim, kernel_size=4, padding='same', strides=1)
        self.bn = keras.layers.BatchNormalization(axis=3)
        self.ReLU = keras.layers.ReLU()

    def call(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.ReLU(self.bn(inputs))
        return inputs

class MyNet(keras.Model):
    def __init__ (self,input_dim=3):
        super(MyNet,self).__init__()
        self.cbr1 = CBR(16)
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.cbr2 = CBR(32)
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2,2))

    def call(self, inputs):
        inputs = self.maxpool1(self.cbr1(inputs))
        inputs = self.maxpool2(self.cbr2(inputs))
        return inputs

model = MyNet(3)
data = tf.random.normal((16,224,224,3))
output = model(data)
print(output.shape)
