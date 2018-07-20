

![](http://p20tr36iw.bkt.clouddn.com/tensorflow_logi.jpg)
<!--more-->

# 神经网络训练Weights&biases

>目的：给出一个函数y=0.1x+0.3,y=x*Weights+biases,然后利用tensorflow把Weights变成0.1,biases变成0.3

## 0.导包
```python
import tensorflow as tf
import numpy as np
import os
```

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 设置TensorFlow输出模式，忽略警告
```
## 1.创建数据
```python
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
```
## 2.创建tensorflow结构
```python
# 定义一个初始值为-1到1的随机数，不断提升接近0.1
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
# 定义一个初始值从0开始，不断提升接近0.3
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases
# 计算预测y与真实y的误差
loss=tf.reduce_mean(tf.square(y-y_data))
# 传播误差
# 建立优化器 反向传递误差的工作就教给optimizer了, 使用的误差传递方法是梯度下降法: Gradient Descent
optimizer=tf.train.GradientDescentOptimizer(0.5) # 0.5表示学习效率，小于1的数
# 利用优化器减少误差，提升参数准确度 (使用 optimizer 来进行参数的更新).
train=optimizer.minimize(loss)
init =tf.global_variables_initializer()
```
## 3.结构激活
```python
sess=tf.Session()
sess.run(init) # Session就像指针一样，指向哪就被激活
```

## 4.开始训练
```python
# 给它201步，0.....200
for step in range(201):
    sess.run(train)
    if step%20==0:

```
![](http://p20tr36iw.bkt.clouddn.com/tensorflow_logi.jpg)

>写在最后(参考文章)：[例子2](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/)
