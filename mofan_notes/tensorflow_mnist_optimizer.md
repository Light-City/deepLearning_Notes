

![](http://p20tr36iw.bkt.clouddn.com/cnn_idea%20-w.png)
<!--more-->

# TensorFlow之MNIST手写体识别任务(二)


## 0.写在前面

1.上节文章：[TensorFlow之MNIST手写体识别任务(一)](http://light-city.me/post/ab5f9384.html)

2.上节文章使用了简单的softmax做数字识别，准确率为92%,本节任务---通过卷积神经网络(Convolutional Neural Network,CNN)来进行手写数字的识别。

3.CNN模型

![](http://p20tr36iw.bkt.clouddn.com/cnn_idea%20-w.png)

convolutional layer1 + max pooling;

convolutional layer2 + max pooling;

fully connected layer1 + dropout;

fully connected layer2 to prediction.

input layer => convolutional layer => pooling layer => convolutional layer => pooling layer => fully connected layer => fully connected layer

## 1.数据导入及图片处理

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
# 解决过拟合的有效手段
keep_prob = tf.placeholder(tf.float32)
'''
把xs的形状变成[-1,28,28,1]，
-1代表先不考虑输入的图片例子多少这个维度，
后面的1是channel的数量，因为我们输入的图片是黑白的，
因此channel是1，例如如果是RGB图像，那么channel就是3。
'''
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]
```
## 2.封装方法

```python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # 前后2个都为1
    # Must have strides[0] = strides[3] = 1
    # 二维的,输入x,W,步长(4个长度的列表)
    # WALID,SAME,2种
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```
## 3.建立卷积层+最大池化层

```python
# 卷积层1
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32]) # 隐藏1层
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

# 卷积层2
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
# 隐藏2层
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64
```
## 4.建立全连接层

```python
# 全连接层1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 全连接层2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```
## 5.预测

```python
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```
## 6.优化

```python
# the error between prediction and real data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
# 使用tf.train.AdamOptimizer()作为我们的优化器进行优化，使我们的cross_entropy最小
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
```
## 7.初始化

```python
# important step
sess.run(tf.global_variables_initializer())
```

## 8.训练

```python
# 注，当训练为1000次时，同上一节相比可以看到testing accuracy提升到了最高1,如下图1
# 当训练20000次后，testing accuracy基本稳定到1,如下图2
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(20)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 100 == 0:
        train_accuracy=compute_accuracy(batch_xs,batch_ys)
        print('step %d, training accuracy %g' % (i, train_accuracy))
        # 或者 train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})
        # 解决Tensorflow Deep MNIST: Resource exhausted: OOM when allocating tensor with shape问题
        testSet = mnist.test.next_batch(50)
        test_accuracy=compute_accuracy(testSet[0], testSet[1])
        print('step %d, testing accuracy %g' % (i, test_accuracy))
        print('-----------------------')
```

![](http://p20tr36iw.bkt.clouddn.com/tensor_mnist_1000.jpg)

![](http://p20tr36iw.bkt.clouddn.com/tensor_train_test.jpg)

## 9.参考文章

>[1.CNN 卷积神经网络 1-3](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-05-CNN3/)

>[2.用 CNN 识别数字](https://blog.csdn.net/aliceyangxi1987/article/details/70787997)

>[3.Tensorflow Deep MNIST: Resource exhausted: OOM when allocating tensor with shape[10000,32,28,28]](https://blog.csdn.net/zsg2063/article/details/74332487)
