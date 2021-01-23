import tensorflow as tf
x = tf.convert_to_tensor(10.)
w = tf.Variable(2.)
b = tf.Variable(3.)
with tf.GradientTape(persistent=True) as tape:
    z = w * x + b
dz_dw = tape.gradient(z,w)
dz_db = tape.gradient(z,b)
print(dz_dw)
print(dz_db)
