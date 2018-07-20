

![](http://p20tr36iw.bkt.clouddn.com/tensor_fig.png)
<!--more-->

# Tensorflow之自编码 Autoencoder (非监督学习)

## 1.可视化解压前后的数字图片
>导包

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
```
>获得数据

```python
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```
>定义相关的Parameter

```python
# Parameter
learning_rate = 0.01
training_epochs = 5 # 五组训练
batch_size = 256
display_step = 1
examples_to_show = 10
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
```

>palceholder hold住数据

```python
X = tf.placeholder("float", [None, n_input])
```

>定义两层实现encode与decode

```python
'''
encode：784->256; 256->128
decode: 128->256; 256->784
'''
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features

weights = {
	'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
	}
biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable(tf.random_normal([n_input])),
	}

# Building the encoder and decoder
def encoder(x):
    '''
    上一层的信号（也就是wx+b算出的结果）要作为下一层的输入，
    但是这个上一层的信号在输入到下一层之前需要一次激活
    f = sigmoid(wx+b)，因为并不是所有的上一层信号
    都可以激活下一层，如果所有的上一层信号都可以激活下一层，
    那么这一层相当于什么都没有做。
    '''
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2
def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    return layer_2
```
>利用方法构建模型

```python
# Construct model
encoder_op = encoder(X) 			# 128 Features
decoder_op = decoder(encoder_op)	# 784 Features

# Prediction通过decode得到y_pred
y_pred = decoder_op	# After
# Targets (Labels) are the input data.
y_true = X			# Before
```

>对比原数据与decode后数据的差异,并选择相应的优化器进行优化

```python
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

>生成图

```python
# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
```
>输出

![](http://p20tr36iw.bkt.clouddn.com/tensor_auto.png)


## 2.可视化聚类图

注:改动代码为:

1.weights/biases以及encoder/decoder方法(主要是增加层)

2.将以上的数字图变为聚类散点图

>完整代码

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Parameter
learning_rate = 0.001
training_epochs = 20 # 五组训练
batch_size = 256
display_step = 1
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)


X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2
weights = {
	'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
	'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4,n_hidden_3])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3,n_hidden_2])),
	'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
	'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
	}
biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
	'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b4': tf.Variable(tf.random_normal([n_input])),
	}

# Building the encoder and decoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    # 为了便于编码层的输出，编码层随后一层不使用激活函数
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                     biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    X=encoder_result[:, 0]
    Y=encoder_result[:, 1]
    T=np.arctan2(X,Y)
    plt.scatter(X, Y, c=T)

    ax=plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()
```
>输出

![](http://p20tr36iw.bkt.clouddn.com/tensor_fig.png)


## 3.参考文章
>[3.1.自编码 Autoencoder (非监督学习)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-11-autoencoder/)

>[3.2.Scatter 散点图](https://morvanzhou.github.io/tutorials/data-manipulation/plt/3-1-scatter/)
