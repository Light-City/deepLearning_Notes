
![](http://p20tr36iw.bkt.clouddn.com/tensorboard.png)
<!--more-->

# Tensorboard可视化(一)

## 1.搭建图纸

>input层开始

```python
# 将xs和ys包含进来，形成一个大的图层，图层名字叫做inputs
with tf.name_scope('inputs'):
    # 为xs指定名称x_input
    xs = tf.placeholder(tf.float32, [None, 1],name='x_input')
    # 为ys指定名称y_input
    ys = tf.placeholder(tf.float32, [None, 1],name='y_input')

```
![](http://p20tr36iw.bkt.clouddn.com/ten_inputs_show.png)
>layer层

```python
# 定义添加神经层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    # 定义大框架名字为layer
    with tf.name_scope('layes'):
        # 框架里面的小部件Weights定义,同时也可以在weights中指定名称W(将会在Weights展开后显示)
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_uniform([in_size,out_size]),name='W')
        # 框架里面的小部件biases定义
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
        # 框架里面的小部件Wx_plus_b定义
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        '''
        activation_function 的话，可以暂时忽略。因为当选择
        用 tensorflow 中的激励函数（activation function）的时候，
        tensorflow会默认添加名称,这个可以在图形呈现后对比两个layer层进行查看
        '''
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs
```
>定义两层

```python
# 定义隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
# 定义输出层
prediction=add_layer(l1,10,1,activation_function=None)
```
![](http://p20tr36iw.bkt.clouddn.com/layer_show.png)

>绘制loss

```python
# 计算预测值prediction与真实值的误差，对两者差的平方求和再取平均
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                   reduction_indices=[1]))
```

![](http://p20tr36iw.bkt.clouddn.com/ten_loss.png)

>绘制train

```python
# 机器学习提升准确率
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 0.1表示学习效率
```
![](http://p20tr36iw.bkt.clouddn.com/ten_train.png)

>收集框架并存储至logs/目录

```python
sess=tf.Session()
writer=tf.summary.FileWriter("logs/",sess.graph)
```
>PyCharm Terminal直接进入项目根目录，运行`tensorboard --logdir=logs`,复制相应的链接至谷歌浏览器你去即可！

![](http://p20tr36iw.bkt.clouddn.com/tensorboard.png)


## 2.参考文章

>[1.Tensorboard 可视化好帮手 1](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-1-tensorboard1/)

>[2.Windows系统下Tensorboard显示空白的问题](https://blog.csdn.net/shanlf/article/details/60589633)

# Tensorboard可视化(二)
## 1.导包
```python
import tensorflow as tf
import numpy as np
```
## 2.make up some data
```python
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
```
## 3.将xs和ys包含进来，形成一个大的图层，图层名字叫做inputs
```python
with tf.name_scope('inputs'):
    # 为xs指定名称x_input
    xs = tf.placeholder(tf.float32, [None, 1],name='x_input')
    # 为ys指定名称y_input
    ys = tf.placeholder(tf.float32, [None, 1],name='y_input')
```
## 4.在 layer 中为 Weights, biases 设置变化图表
```python
# add_layer多加一个n_layer参数(表示第几层)
def add_layer(inputs ,
              in_size,
              out_size,n_layer,
              activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
         # 对weights进行绘制图标
         with tf.name_scope('weights'):
              Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')
              tf.summary.histogram(layer_name + '/weights', Weights)
          # 对biases进行绘制图标
         with tf.name_scope('biases'):
              biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
              tf.summary.histogram(layer_name + '/biases', biases)
         with tf.name_scope('Wx_plus_b'):
              Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)
         if activation_function is None:
            outputs=Wx_plus_b
         else:
            outputs= activation_function(Wx_plus_b)
         # 对outputs进行绘制图标
         tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
```
## 5.修改隐藏层与输出层
```python
# 由于我们对addlayer 添加了一个参数, 所以修改之前调用addlayer()函数的地方. 对此处进行修改:
# add hidden layer
l1= add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output  layer
prediction= add_layer(l1, 10, 1, n_layer=2, activation_function=None)
```
![](http://p20tr36iw.bkt.clouddn.com/ten_histograms.jpg)
![](http://p20tr36iw.bkt.clouddn.com/ten_distributions.jpg)
## 6.设置loss的变化图
```python
#  loss是在tesnorBorad 的event下面的, 这是由于我们使用的是tf.scalar_summary() 方法.
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)  # tensorflow >= 0.12
```
![](http://p20tr36iw.bkt.clouddn.com/ten_scalars.jpg)
## 7.给所有训练图合并
```python
# 机器学习提升准确率
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 0.1表示学习效率

# 初始化
sess= tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph) #
sess.run(tf.global_variables_initializer())
```
## 8.训练数据
```python
for i in range(1000):
   sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
   if i%50 == 0:
      rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
      writer.add_summary(rs, i)
```
![](http://p20tr36iw.bkt.clouddn.com/ten_graphs.jpg)

## 9.问题

若在浏览器输入相应的链接，没有显示，试试关闭防火墙即可解决！

## 10.参考文章
>[Tensorboard 可视化好帮手 2](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-2-tensorboard2/)
