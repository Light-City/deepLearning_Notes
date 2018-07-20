

![](http://p20tr36iw.bkt.clouddn.com/MNIST.png)
<!--more-->

# TensorFlow之MNIST手写体识别任务(一)

## 1.MNIST数据集
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
```

#### 通过以上代码会自动下载和安装这个数据集，该数据集共分为三部分，训练数据集(mnist.train)、测试数据集和验证数据集。一般来说，训练数据集是用来训练模型，验证数据集可以检验所训练出来的模型的正确性和是否过拟合，测试集是不可见的（相当于一个黑盒），但我们最终的目的是使得所训练出来的模型在测试集上的效果（这里是准确性）达到最佳。

#### 每一个MNIST数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。训练数据集和测试数据集都包含xs和ys，比如训练数据集的图片是 mnist.train.images ，训练数据集的标签是 mnist.train.labels。

#### 在MNIST训练数据集中，mnist.train.images 是一个形状为 [60000, 784] 的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间。相对应的MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。这里用One-hot vector来表述label值，vector的长度为label值的数目，vector中有且只有一位为1，其他为0，标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])。mnist.train.labels 是一个 [60000, 10] 的数字矩阵。

## 2.Softmax Regression模型
#### Softmax Regression大致分为两步：
```python
Step 1: add up the evidence of our input being in certain classes;
Step 2: convert that evidence into probabilities.
```

<a href="http://www.codecogs.com/eqnedit.php?latex=evidence_i=\sum_jW_{i,j}*x_j&plus;b_i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?evidence_i=\sum_jW_{i,j}*x_j&plus;b_i"/></a>
>对于softmax回归模型可以用下面的图解释，对于输入的xs加权求和，再分别加上一个偏置量，最后再输入到softmax函数中

![](http://www.tensorfly.cn/tfdoc/images/softmax-regression-scalargraph.png)
>我们可以得到

![](http://www.tensorfly.cn/tfdoc/images/softmax-regression-scalarequation.png)
>我们也可以用向量表示这个计算过程：用矩阵乘法和向量相加

![](http://www.tensorfly.cn/tfdoc/images/softmax-regression-vectorequation.png)

## 3.定义添加神经层的函数
```python
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_uniform([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
```
## 4.Softmax Regression的程序实现
```python
import tensorflow as tf
>输入任意数量的MINIST图像

'''
我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
（这里的None表示此张量的第一个维度可以是任何长度的。）
'''
xs=tf.placeholder(tf.float32,[None,784])

prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
```
## 5.模型的训练---交叉熵函数(分类算法：Cross entropy loss)
![](http://www.tensorfly.cn/tfdoc/images/mnist10.png)
#### y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)
```python
'''
分类问题的目标变量是离散的，而回归是连续的数值。
分类问题，用 onehot + cross entropy
training 过程中，分类问题用 cross entropy，回归问题用 mean squared error。
training 之后，validation / testing 时，使用 classification error，更直观，而且是我们最关注的指标。
'''
ys=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize((cross_entropy))
```
## 6.Session初始化
```python
sess=tf.Session()
sess.run(tf.global_variables_initializer())
```
## 7.模型训练

```python
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
```
## 8.模型的评价
#### 怎样评价所训练出来的模型？显然，我们可以用图片预测类别的准确率

```python
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    # 利用tf.argmax()函数来得到预测和实际的图片label值，再用一个tf.equal()函数来判断预测值和真实值是否一致
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    '''correct_prediction是一个布尔值的列表，例如 [True, False, True, True]。
    可以使用tf.cast()函数将其转换为[1, 0, 1, 1]，以方便准确率的计算
    （以上的是准确率为0.75）。
    '''
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result
```
#### 我们来获取模型在测试集上的准确率
```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```
## 9.完整代码及运行结果
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
准备数据(MNIST库---手写数字库，数据中包含55000张训练图片，
每张图片的分辨率是28*28，所以我们训练网络输入应该是28*28=784个像素数据)
'''
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
# 搭建网络
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
# 定义添加神经层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_uniform([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

# loss(分类算法：Cross entropy loss)
'''
分类问题的目标变量是离散的，而回归是连续的数值。
分类问题，用 onehot + cross entropy
training 过程中，分类问题用 cross entropy，回归问题用 mean squared error。
training 之后，validation / testing 时，使用 classification error，更直观，而且是我们最关注的指标。
'''
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize((cross_entropy))
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(
            mnist.test.images,mnist.test.labels),end="->")
```

![](http://p20tr36iw.bkt.clouddn.com/ten_res.png)
