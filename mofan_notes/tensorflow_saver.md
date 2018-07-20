

![](http://p20tr36iw.bkt.clouddn.com/saver.jpg)
<!--more-->
# TensorFlow之Saver保存读取

## 0.写在前面
我们在搭建号一个神经网络，训练好后，肯定想保存起来，用于再次加载。本文通过Tensorflow中的saver保存和加载变量

## 1.保存

```python
import tensorflow as tf

# Save to file
# remember to define the same dtype shape when restore
W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init=tf.global_variables_initializer()

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,"my_net/save_net.ckpt")
    print("Save to path:",save_path)

```
![](http://p20tr36iw.bkt.clouddn.com/tensorflow_save.jpg)

## 2.读取

```python
import numpy as np
# restore variable
# redefine the same shape and same type for your variables
W=tf.Variable(np.arange(6).reshape(2,3),dtype=tf.float32,name='weights')
b=tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32,name='biases')
# not need init step
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))
```
![](http://p20tr36iw.bkt.clouddn.com/tensor_weights_biases.jpg)

## 3.参考文章
>[Saver 保存读取](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-06-save/)
