import tensorflow as tf
import input_data


#导入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#占位符placeholder 这里的None表示此张量的第一个维度可以是任何长度的。
x = tf.placeholder("float", [None, 784])

#Variable可以用于计算输入值，也可以在计算中被修改。
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#[n, 784] * [784, 10] + [10]
y = tf.nn.softmax(tf.matmul(x,W) + b)
print(y)

#训练模型

#继续占位符 y'
y_ = tf.placeholder("float", [None,10])
#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#创建初始化变量
init = tf.initialize_all_variables()
#Session里面启动我们的模型，并且初始化变量
sess = tf.Session()
sess.run(init)

#迭代1000次
for i in range(1000):
  #随机梯度下降训练，每次随机抓取100个数据点
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#评估模型 argmax 1标签所在的索引就是样本的数值 y_为真实值， y为预测值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#[ ] 计算正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))