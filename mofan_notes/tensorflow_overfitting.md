

![](http://p20tr36iw.bkt.clouddn.com/Overfitting.png)
<!--more-->
# TensorFlow之Overfitting

## 1.回归分类的过拟合
如下图-比如：我们希望机器学习模型出来是一条直线，此时的蓝线与数据的总误差可能是10，但机器为了过去纠结降低误差，就会几乎经过了每一个数据点，此时成为曲线，误差为1，可这并不是我们想要的，即机器学习的自负就体现出来了。

![](http://p20tr36iw.bkt.clouddn.com/ten_overfitting.png)

## 2.解决办法
![](http://p20tr36iw.bkt.clouddn.com/ten_over.png)
>方法一：增加数据量，大部分过拟合产生的原因是因为数据量太少了，如果有成千上万的数据，红线也会慢慢被拉直，变得没那么扭曲。

![](http://p20tr36iw.bkt.clouddn.com/ten_regul.png)
>方法二：运用正规化/l1,l1 regularization等(l3,l4...分别将后面的变为3次方,4次方.....)

![](http://p20tr36iw.bkt.clouddn.com/ten_dropout.png)
>方法三：专门用在神经网络的正规化的方法，叫做dropout.我们随机忽略掉一些神经元和神经联结 , 是这个神经网络变得“不完整”. 用一个不完整的神经网络训练一次，到第二次再随机忽略另一些, 变成另一个不完整的神经网络. 有了这些随机 drop 掉的规则, 我们可以想象其实每次训练的时候, 我们都让每一次预测结果都不会依赖于其中某部分特定的神经元. 像l1, l2正规化一样, 过度依赖的 W , 也就是训练参数的数值会很大, l1, l2会惩罚这些大的 参数. Dropout 的做法是从根本上让神经网络没机会过度依赖.
## 3.Dropout 缓解过拟合

### 3.1导包

```python
import tensorflow as tf
from sklearn.datasets import load_digits
'''
如果使用cross_validation，则报错 DeprecationWarning:
This module was deprecated in version 0.18 in favor of
the model_selection module into which all the refactored classes
and functions are moved. Also note that the interface of the new CV
iterators are different from that of this module. This module will be
removed in 0.20."This module will be removed in 0.20.", DeprecationWarning
'''
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelBinarizer
```
### 3.2建立 dropout 层

>准备数据

```python
# load data
digits=load_digits()
# 从0-9的图片data
X=digits.data
# 1表示为[0,1.....0]这种Binary形式
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)


```
>定义placeholder

```python
# 这里的keep_prob是保留概率，即我们要保留的结果所占比例，它作为一个placeholder，在run时传入， 当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用。
keep_prob=tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64]) # 8*8
ys = tf.placeholder(tf.float32, [None, 10]) # 0-9

```

>定义add_layer方法，并在里面添加dropout功能

```python

# 在add_layer方法里面添加dropout功能
# 定义添加神经层的函数
def add_layer(inputs ,
              in_size,
              out_size,n_layer,
              activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        # dropout功能
        Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            # tf.histogram_summary(layer_name+'/outputs',outputs)
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
```

>添加隐藏层与输出层

```python

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)
```
>loss函数（即最优化目标函数）选用交叉熵函数。

```python
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零。
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
```
>train方法（最优化算法）采用梯度下降法。

```python
train_step=tf.train.GradientDescentOptimizer(0.6).minimize((cross_entropy))
```
>

```python
'''
 tf.summary.FileWriter()  将上面‘绘画’出的图保存到一个目录中，以方便后期在浏览器中可以浏览。 这个方法中的第二个参数需要使用sess.graph . 因此我们需要把这句话放在获取session的后面。 这里的graph是将前面定义的框架信息收集起来，然后放在logs/tain与logs/test目录下面。
'''
sess=tf.Session()
# 合并所有的summary
merged=tf.summary.merge_all()
```
Creates a `FileWriter` and an event file，This event file will contain
 `Event` protocol buffers constructed when you call one of the following functions: `add_summary()`, `add_session_log()`,`add_event()`, or `add_graph()`.
```
train_writer=tf.summary.FileWriter("logs/train",sess.graph)
test_write=tf.summary.FileWriter("logs/test",sess.graph)
```
>初始化

```python
sess.run(tf.global_variables_initializer())
```

### 3.3训练

```python
for i in range(500):
    # 如果想drop掉40%，那么keep_prob就得为0.6，表示保持60%的概率不被drop掉的。
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
    if i%50==0:
        # record loss,不drop掉任何东西，即keep_prob为1
        train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result=sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_write.add_summary(test_result,i)
```
橙线是 test 的误差, 蓝线是 train 的误差.
训练中keep_prob=1时，就可以暴露出overfitting问题。
![](http://p20tr36iw.bkt.clouddn.com/tensor_loss.png)

keep_prob=0.5时，dropout就发挥了作用。 我们可以两种参数分别运行程序，对比一下结果。
![](http://p20tr36iw.bkt.clouddn.com/tensor_dropout0.5.png)

keep_prob=0.4时，dropout就发挥了作用。 我们可以两种参数分别运行程序，对比一下结果。
![](http://p20tr36iw.bkt.clouddn.com/tensor_dropout0.6.png)

keep_prob=0.3时，dropout就发挥了作用。 我们可以两种参数分别运行程序，对比一下结果。
![](http://p20tr36iw.bkt.clouddn.com/tensorflow0.7.png)

当keep_prob为0.2及以下时，也就是dropout>70%后，会报错!合适的dropout保持在50%左右最好。

## 4.参考文章

>[1.什么是过拟合 (Overfitting)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-02-A-overfitting/)

>[2.Dropout 解决 overfitting](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-02-dropout/)
