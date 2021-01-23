from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
# image = Image.open('./bug1.jpeg')
# image.show()
# image = np.array(image)
# print(image.shape)
#--------------------------------
# images = tf.io.gfile.glob('./*.jpeg')
# image = tf.io.read_file('./bug1.jpeg')
# image = tf.image.decode_jpeg(image,channels=3)
# print(image.shape,type(image))
#--------------------------------
def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3, ratio=1)
    image = tf.image.resize(image, [256, 256])  # 统一图片大小
    image = tf.cast(image, tf.float32)  # 转换类型
    image = image / 255  # 归一化
    return image
images = tf.io.gfile.glob('./*.jpeg')
dataset = tf.data.Dataset.from_tensor_slices(images)
AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = dataset.map(read_image,num_parallel_calls=AUTOTUNE)
dataset = dataset.shuffle(1).batch(1)
for a in dataset.take(2):
    print(a.shape)
#-----------
# dataset = tf.data.TFRecordDataset(['bug1.jpeg','bug2.jpeg'])
# image = tf.io.read_file(input_path)