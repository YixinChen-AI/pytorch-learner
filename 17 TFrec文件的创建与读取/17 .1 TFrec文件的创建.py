import tensorflow as tf
import glob
# 先记录一下要保存的tfrec文件的名字
tfrecord_file = './train.tfrec'
# 获取指定目录的所有以jpeg结尾的文件list
images = glob.glob('./*.jpeg')
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename in images:
        image = open(filename, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {  # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
            'float':tf.train.Feature(float_list=tf.train.FloatList(value=[1.0,2.0])),
            'name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(filename)]))
        }
        # tf.train.Example 在 tf.train.Features 外面又多了一层封装
        example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
        writer.write(example.SerializeToString())  # 将 Example 序列化并写入 TFRecord 文件