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
    def __init__ (self):
        super(MyNet,self).__init__()
        self.cbr1 = CBR(16)
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.cbr2 = CBR(32)
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.cbr3 = CBR(16)
        self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.cbr4 = CBR(1)

    def call(self, inputs):
        inputs = self.maxpool1(self.cbr1(inputs))
        inputs = self.maxpool2(self.cbr2(inputs))
        inputs = self.maxpool3(self.cbr3(inputs))
        inputs = self.cbr4(inputs)
        return inputs

model = MyNet()
model.build((16,8,8,3))
print(model.summary())


# model.save('save_model.h5')
# new_model = keras.models.load_model('save_model.h5')
model.save_weights('model_weight')
new_model = MyNet()
new_model.load_weights('model_weight')
# 看一下原来的模型和载入的模型预测相同的样本的输出
test = tf.ones((1,8,8,3))
prediction = model.predict(test)
new_prediction = new_model.predict(test)
print(prediction,new_prediction)
