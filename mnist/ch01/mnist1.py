import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

shape = 28 * 28
# 定义输入图片数据的占位符
x = tf.placeholder(tf.float32, [None, shape])

# 定义矩阵 W
W = tf.Variable(tf.zeros([shape, 10]))
# 定义偏置 b
b = tf.Variable(tf.zeros(10))

# 定义y的计算方式
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义正确输出结果的占位符
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义损失函数,由于是批量传入数据，所以计算的是批量数据的平均损失函数，这里注意，tf.reduce_sum前是-
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 定义梯度下降方法
train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 计算准确的数值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))