
![](http://p20tr36iw.bkt.clouddn.com/tensorflow_gd.png)
<!--more-->

# 用 Tensorflow 可视化梯度下降

>导包

```python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```
>定义真实数据与测试数据

```python
LR=.1
REAL_PARAMS=[1.2,2.5]
INT_PARAMS=[[5,4],
            [5,1],
            [2,4.5]][0]

```
>定义真实与测试方程

```python
x=np.linspace(-1,1,200,dtype=np.float32)
'''
lambda方程
def y_fun(a,b):
    return a*x+b
'''
y_fun = lambda a, b: a * x + b
tf_y_fun = lambda a, b: a * x + b

'''
1.*号在定义函数参数时，传入函数的参数会转换成元组，
如果 *号在调用时则会把元组解包成单个元素。
2.**号在定义函数参数时，传入函数的参数会转换成字典，
如果 **号在调用时则会把字典解包成单个元素。
'''
'''
numpy中有一些常用的用来产生随机数的函数，例如：randn()与rand()。
numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中
'''

# sigma * np.random.randn(...) + mu
noise=np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise
# plt.scatter(x,y)
# plt.show()
```
>利用梯度下降方法进行优化

```python
a,b=[tf.Variable(initial_value=p,dtype=tf.float32) for p in INT_PARAMS]
pred=tf_y_fun(a,b)
mse=tf.reduce_mean(tf.square(y-pred))
train_op=tf.train.GradientDescentOptimizer(LR).minimize(mse)
a_list,b_list,cost_list=[],[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(400):
        a_,b_,mse_=sess.run([a,b,mse])
        a_list.append(a_);b_list.append(b_);cost_list.append(mse_)
        result,_=sess.run([pred,train_op])
```
>可视化直线图

```python
# visualization codes:
print('a=', a_, 'b=', b_)
plt.figure(1)
plt.scatter(x, y, c='b')    # plot data
plt.plot(x, result, 'r-', lw=2)   # plot line fitting

```
>可视化3D图形

```python
# 3D cost figure
fig = plt.figure(2); ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  # parameter space
'''
a = [1,2,3]
b = [4,5,6]
zipped = zip(a,b)     # 打包为元组的列表
输出：[(1, 4), (2, 5), (3, 6)]
a=array([[1,2],[3,4],[5,6]])  ###此时a是一个array对象
a
输出：array([[1,2],[3,4],[5,6]])
a.flatten()
输出：array([1,2,3,4,5,6])
flatten()即返回一个折叠成一维的数组。但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的
'''
cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
# 绘制点
ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # initial parameter place
ax.set_xlabel('a'); ax.set_ylabel('b')
# 绘制线条
ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)    # plot 3D gradient descent
plt.show()

```


>output

将a,b与REAL_PARAMS对比，查看相近度！

```
a= 1.1898687 b= 2.506877
```
![](http://p20tr36iw.bkt.clouddn.com/tensorflow_line.png)

![](http://p20tr36iw.bkt.clouddn.com/tensorflow_gd.png)

>参考文章

[用 Tensorflow 可视化梯度下降](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-15-tf-gradient-descent/)
