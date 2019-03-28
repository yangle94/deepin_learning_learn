# coding: utf-8

# 两层卷积网络

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28 * 28])
y_ = tf.placeholder(tf.float32, [None, 10])

# 将784的转换为28*28
# 省略最后的1是否可以   image = tf.reshape(x, shape=[-1, 28, 28，1])
image = tf.reshape(x, shape=[-1, 28, 28, 1])


# 根据shape创建矩阵
def weight_variable(shape):
    # 正太分布，标差0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 创建偏置项
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积操作
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化操作，ksize代表池化大小，strides代表不同方向步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一次卷积、池化操作
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二次卷积、池化操作
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接部分,将第二次卷积池化操作的变为一维    ? 为什么转换成1024维的数组
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_float = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_float, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

# 全连接，第二次，将1024转换为10个数字分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

# 定义损失函数
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)

# 定义梯度下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义准确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_, 1), tf.arg_max(y_conv, 1)), tf.float32))

# 创建session， 初始化变量
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练20000步
for i in range(2000):
    train_batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    sess.run(train_step, feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})

train_accuracy_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print("training accuracy %g" % train_accuracy_test)