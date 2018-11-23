import tensorflow as tf
import numpy as np


coefficients = np.array([[1.], [-10.], [25.]])

x = tf.placeholder(tf.float32, [3, 1])
w = tf.Variable(0, dtype=tf.float32)
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(train, feed_dict={x:coefficients})
    print(session.run(w))