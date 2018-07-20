

![](http://p20tr36iw.bkt.clouddn.com/tensorflow_learn.jpg)
<!--more-->

# TensorFlow学习

## 1.Session会话控制(两种打开模式)

>定义矩阵

```python
matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],
                     [2]])
```

>两矩阵相乘

```python
product=tf.matmul(matrix1,matrix2) # matrix mul
```

>Session会话控制方法一

```python
sess=tf.Session()
result=sess.run(product)
print(result) # [[12]]
sess.close()
```

>Session会话控制方法二

```python
# 好处就是不需要写sess.close，因为以下代码在with语句中运行，运行完毕，自动关闭了
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2) # [[12]]
```
## 2.Tensorflow使用Variable
>写在前面

```python
在 Tensorflow 中，定义了某字符串是变量，它才是变量
```
>定义变量与常量

```python
state=tf.Variable(0,name='counter')
print(state.name) # counter:0
one=tf.constant(1)
```
>变量与常量做加法运算

```python
new_value=tf.add(state,one)
```
>更新state值

```python
update=tf.assign(state,new_value)
```
>变量初始化!!!

```python
# 如果定义变量一定要用这个！
# init=tf.initialize_all_variables() 即将被废除
init=tf.global_variables_initializer()
# 注意：到这里变量还是没有被激活，需要在下面 sess 里, sess.run(init) , 激活 init
```
>激活变量

```python
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

注意：直接 print(state) 不起作用！！一定要把 sess 的指针指向 state 再进行 print 才能得到想要的结果！
```


## 3.Placeholder 传入值
>写在前面


```python
'''
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(),
然后以这种形式传输数据stat.run(***,feed_dict(key:value,key1:value.....))
'''
```

>定义两个placeholder

```python
# 在Tensorflow中需要定义placeholder的type,一般为type32形式
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
```

>mul=multiply是将input1和input2做乘法运算

```python
output=tf.multiply(input1,input2)
```
>外部传如data，并输出结果

```python
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[2],input2:[3.0]}))
```

## 4.激励函数(activate function)

>激励函数运行时激活神经网络中某一部分神经元。

>将激活信息向后传入下一层的神经系统。

>激励函数的实质是非线性方程。

## 5.定义添加神经层的函数
>写在前面

```python
定义添加神经层的函数def add_layer(),它有四个参数：
输入值、输入的大小、输出的大小和激励函数，
我们设定默认的激励函数是None
```

>定义weights和biases

```python
# 因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵
Weights=tf.Variable(tf.random_uniform([in_size,out_size]))
# biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1
biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
```
>激励函数处理

```python
# 当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，不为None时，就把Wx_plus_b传到activation_function()函数中得到输出
if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
```

>返回输出

```python
return outputs
```

>完整函数

```python
import tensorflow as tf
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
```
## 6.建造神经网络
>导入包numpy

```python
import numpy as np
```

>定义添加神经层的函数

```python
import tensorflow as tf
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

>将一个有300个元素的一维数组转换成1列300行的矩阵形式（列向量）

```python
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
```
>噪点，没有按照函数走,这样看起来会更像真实情况，其中0.05表示方差

```python
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float64)
y_data=np.square(x_data)-0.5+noise
```
>接下来，开始定义神经层。 通常神经层都包括输入层、隐藏层和输出层。这里的输入层只有一个属性， 所以我们就只有一个输入；隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元； 输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。 所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。

```python
# 定义隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
# 定义输出层
predition=add_layer(l1,10,1,activation_function=None)

# 计算预测值predition与真实值的误差，对两者差的平方求和再取平均
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),
                   reduction_indices=[1]))
# 机器学习提升准确率
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 0.1表示学习效率

# 初始化
init=tf.global_variables_initializer()
# 激活变量
sess=tf.Session()
sess.run(init)
```
>训练

```python
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # 每50步我们输出一下机器学习的误差
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
```
>运行

![](http://p20tr36iw.bkt.clouddn.com/numpy_res.jpg)

## 7.matplotlib 可视化
>参见前文

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
# 将一个有300个元素的一维数组转换成1列300行的矩阵形式（列向量）
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
# 噪点，没有按照函数走,这样看起来会更像真实情况，其中0.05表示方差
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float64)
y_data=np.square(x_data)-0.5+noise
# 输入层1，隐藏层10，输出层1
# 这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 定义隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
# 定义输出层
prediction=add_layer(l1,10,1,activation_function=None)
# 计算预测值prediction与真实值的误差，对两者差的平方求和再取平均
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                   reduction_indices=[1]))
# 机器学习提升准确率
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 0.1表示学习效率
# 初始化
init=tf.global_variables_initializer()
# 激活变量
sess=tf.Session()
sess.run(init)
```

>绘制散点图

```python
# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion() # 连续显示
plt.show()
```
>显示预测数据

```python
# 每隔50次训练刷新一次图形，用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.1s。
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction r-表示红色实线,lw表示线宽
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
```
>问题：红色实线条不显示，解决办法：取消matplotlib默认输出到sciview

```python
取消Settings>Tools>Python Scientific>Show plots in toolwindow勾选项
```

![](http://p20tr36iw.bkt.clouddn.com/tf_learn.gif)


## 8.参考文章
#### [1.Tensorflow简介](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/)
#### [2.新版pycharm中，取消matplotlib默认输出到sciview](https://blog.csdn.net/chengyu_whu/article/details/80493477)
