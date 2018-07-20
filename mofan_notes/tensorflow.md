

![](http://p20tr36iw.bkt.clouddn.com/tensorflow.jpg)
<!--more-->

# Win10、PyCharm、Cuda9.2、cuDNN7.1、tensorflow-gpu1.8部署

### 注：本文主要讲tensorflow-gpu 1.8的安装及测试

## 一、Anaconda的envs下新建一个python3环境，将Tensor flow安装在这里面
>直达站点

### [Win10上Anaconda的python2与python3切换](http://light-city.me/post/4683d07b.html)
## 二、关于Cuda 9.2与cuDNN 7.1的安装
>直接[Cuda官网下载](https://developer.nvidia.com/cuda-toolkit-archive)|[cuDNN官网下载(需注册)](https://developer.nvidia.com/rdp/cudnn-download)
## 三、关于Cuda 9.2与cuDNN 7.1安装环境配置及测试
### 1.Cuda 9.2配置
>一路安装成功后，dos输入nvcc -V，显示如下信息表示安装成功

![](http://p20tr36iw.bkt.clouddn.com/nvcc.jpg)
>默认情况

#### 安装好后，默认情况下，系统变量会多出CUDA_PATH和CUDA_PATH_V9_2两个环境变量。

>添加系统变量

```python
CUDA_SDK_PATH = C:\ProgramData\NVIDIACorporation\CUDA Samples\v9.2
CUDA_LIB_PATH = %CUDA_PATH%\lib\x64
CUDA_BIN_PATH = %CUDA_PATH%\bin
CUDA_SDK_BIN_PATH = %CUDA_SDK_PATH%\bin\win64
CUDA_SDK_LIB_PATH = %CUDA_SDK_PATH%\common\lib\x64
```

### 2.cuDNN

>解压cudnn-9.2-windows10-x64-v7.1.zip，将文件夹里的内容拷贝到CUDA的安装目录并覆盖相应的文件夹，CUDA默认安装目录：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2

## 四、tensorflow-gpu 1.8
### 1.安装
### 注：目前tensorflow还不能支持cuda9.2，所以只能通过源码编译或者别人编译好的安装包安装。
>[下载地址](https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.8.0/py36/GPU/cuda92cudnn71sse2)
下载这个wheel包后，使用pip install .....whl进行安装

### 2.测试
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
```

### 3.问题
>警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

```python
#解决
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
>FutureWarning: Conversion of the second argument of issubdtype from float to np.floating is deprecated

```python
h5py新版本对numpy1.4版本的兼容错误
更新numpy与h5py问题解决
```

## 五、参考文章
[1.警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA](https://blog.csdn.net/hq86937375/article/details/79696023)

[2.成功解决Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2](https://blog.csdn.net/qq_41185868/article/details/79127838)

[3.Win10 64 位Tensorflow-gpu安装（VS2017+CUDA9.2+cuDNN7.1.4+python3.6.5）](https://blog.csdn.net/wwtor/article/details/80603296)

[4.导入tensorflow错误：FutureWarning:Conversion of the second argument of issubdtype from \`float\`省略](https://blog.csdn.net/u014561933/article/details/80156091)
