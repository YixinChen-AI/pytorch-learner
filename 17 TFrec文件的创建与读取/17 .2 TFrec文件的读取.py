import tensorflow as tf

dataset = tf.data.TFRecordDataset('./train.tfrec')

def decode(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'float': tf.io.FixedLenFeature([1, 2], tf.float32),
        'name': tf.io.FixedLenFeature([], tf.string)
    }
    feature_dict = tf.io.parse_single_example(example, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])  # 解码 JEPG 图片
    return feature_dict

dataset = dataset.map(decode).batch(4)
for i in dataset.take(1):
    print(i['image'].shape)
    print(i['label'].shape)
    print(i['float'].shape)
    print(bytes.decode(i['name'][0].numpy()))


