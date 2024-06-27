#!/usr/bin/env python
# coding: utf-8

# # 网络参数
# 
# [![在线运行](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_modelarts.png)](https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svcjEuNy90dXRvcmlhbHMvemhfY24vYWR2YW5jZWQvbmV0d29yay9taW5kc3BvcmVfcGFyYW1ldGVyLmlweW5i&imageid=9d63f4d1-dc09-4873-b669-3483cea777c0)&emsp;[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.7/tutorials/zh_cn/advanced/network/mindspore_parameter.ipynb)&emsp;[![下载样例代码](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_download_code.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.7/tutorials/zh_cn/advanced/network/mindspore_parameter.py)&emsp;[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.7/tutorials/source_zh_cn/advanced/network/parameter.ipynb)

# MindSpore提供了关于变量、网络相关参数的初始化模块，用户可以通过封装算子来调用字符串、Initializer子类或自定义Tensor等方式完成对网络参数进行初始化。
# 
# 下面图中蓝色表示具体的执行算子，绿色的表示张量Tensor，张量作为神经网络模型中的数据在网络中不断流动，主要包括网络模型的数据输入，算子的输入输出数据等；红色的为变量Parameter，作为网络模型或者模型中算子的属性，及其反向图中产生的中间变量和临时变量。
# 
# ![parameter.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/advanced/network/images/parameter.png)
# 
# 本章主要介绍数据类型`dtype`、变量`Parameter`、变量元组`ParameterTuple`、网络的初始化方法和网络参数更新。
# 
# ## 数据类型 dtype
# 
# MindSpore张量支持不同的数据类型`dtype`，包含int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64、bool_，与NumPy的数据类型一一对应。详细的数据类型支持情况请参考：[mindspore.dtype](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.dtype.html#mindspore.dtype)。
# 
# 在MindSpore的运算处理流程中，Python中的int数会被转换为定义的int64类型，float数会被转换为定义的float32类型。
# 
# 以下代码，打印MindSpore的数据类型int32。

# In[1]:


from mindspore import dtype as mstype

data_type = mstype.int32
print(data_type)


# ### 数据类型转换接口
# 
# MindSpore提供了以下几个接口，实现与NumPy数据类型和Python内置的数据类型间的转换。
# 
# - `dtype_to_nptype`：将MindSpore的数据类型转换为NumPy对应的数据类型。
# - `dtype_to_pytype`：将MindSpore的数据类型转换为Python对应的内置数据类型。
# - `pytype_to_dtype`：将Python内置的数据类型转换为MindSpore对应的数据类型。
# 
# 以下代码实现了不同数据类型间的转换，并打印转换后的类型。

# In[2]:


from mindspore import dtype as mstype

np_type = mstype.dtype_to_nptype(mstype.int32)
ms_type = mstype.pytype_to_dtype(int)
py_type = mstype.dtype_to_pytype(mstype.float64)

print(np_type)
print(ms_type)
print(py_type)


# ## 变量 Parameter
# 
# MindSpore的变量（[Parameter](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter)）表示在网络训练时，需要被更新的参数。例如，在前行计算的时候最常见的`nn.conv`算子的变量有权重`weight`和偏置`bias`；在构建反向图和反向传播计算的时候，会产生很多中间变量，用于暂存一阶梯度信息、中间输出值等。
# 
# ### 变量初始化
# 
# 变量`Parameter`的初始化方法有很多种，可以接收`Tensor`、`Initializer`等不同的数据类型。
# 
# - `default_input`：为输入数据，支持传入`Tensor`、`Initializer`、`int`和`float`四种数据类型；
# - `name`：可设置变量的名称，用于在网络中区别于其他变量；
# - `requires_grad`：表示在网络训练过程，是否需要计算参数梯度，如果不需要计算参数梯度，将`requires_grad`设置为`False`。
# 
# 下面的示例代码中，使用`int`或`float`数据类型直接创建Parameter：

# In[1]:


from mindspore import Parameter

x = Parameter(default_input=2.0, name='x')
y = Parameter(default_input=5.0, name='y')
z = Parameter(default_input=5, name='z', requires_grad=False)

print(type(x))
print(x, "value:", x.asnumpy())
print(y, "value:", y.asnumpy())
print(z, "value:", z.asnumpy())


# 下面示例代码中，使用MindSpore的张量`Tensor`创建Parameter：

# In[2]:


import numpy as np
from mindspore import Tensor

my_tensor = Tensor(np.arange(2 * 3).reshape((2, 3)))
x = Parameter(default_input=my_tensor, name="tensor")

print(x)


# 下面示例代码中，使用`Initializer`创建Parameter：

# In[6]:


from mindspore.common.initializer import initializer as init
from mindspore import dtype as mstype

x = Parameter(default_input=init('ones', [1, 2, 3], mstype.float32), name='x')
print(x)


# ### 属性
# 
# 变量`Parameter`的默认属性有变量名称`name`、形状`shape`、数据类型`dtype`和是否需要进行求导`requires_grad`。
# 
# 下例通过`Tensor`初始化一个变量`Parameter`，并获取变量`Parameter`的相关属性。示例代码如下：

# In[4]:


my_tensor = Tensor(np.arange(2 * 3).reshape((2, 3)))
x = Parameter(default_input=my_tensor, name="x")

print("x: ", x)
print("x.data: ", x.data)


# ### 变量操作
# 
# 1. `clone`：克隆变量张量`Parameter`，克隆完成后可以给新的变量`Parameter`指定新的名称。

# In[9]:


x = Parameter(default_input=init('ones', [1, 2, 3], mstype.float32))
x_clone = x.clone()
x_clone.name = "x_clone"

print(x)
print(x_clone)


# 2. `set_data`：修改变量`Parameter`的数据或形状`shape`。
# 
# 其中，`set_data`方法有`data`和`slice_shape`两种入参。`data`表示变量`Parameter`新传入的数据；`slice_shape`表示是否修改变量`Parameter`的形状`shape`，默认为False。

# In[11]:


x = Parameter(Tensor(np.ones((1, 2)), mstype.float32), name="x", requires_grad=True)
print(x, x.asnumpy())

y = x.set_data(Tensor(np.zeros((1, 2)), mstype.float32))
print(y, y.asnumpy())

z = x.set_data(Tensor(np.ones((1, 4)), mstype.float32), slice_shape=True)
print(z, z.asnumpy())


# 3. `init_data`：并行场景下存在参数的形状发生变化的情况，用户可以调用`Parameter`的`init_data`方法得到原始数据。

# In[9]:


x = Parameter(Tensor(np.ones((1, 2)), mstype.float32), name="x", requires_grad=True)

print(x.init_data(), x.init_data().asnumpy())


# ### 变量参数更新
# 
# MindSpore提供了网络参数更新功能，使用`nn.ParameterUpdate`可对网络参数进行更新，其输入的参数类型必须为张量，且张量`shape`需要与原网络参数`shape`保持一致。
# 
# 更新网络的权重参数示例如下：

# In[14]:


import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, Tensor

# 构建网络
network = nn.Dense(3, 4)

# 获取网络的权重参数
param = network.parameters_dict()['weight']
print("Parameter:\n", param.asnumpy())

# 更新权重参数
update = nn.ParameterUpdate(param)
weight = Tensor(np.arange(12).reshape((4, 3)), mstype.float32)
output = update(weight)
print("Parameter update:\n", output)


# ## 变量元组 Parameter Tuple
# 
# 变量元组[ParameterTuple](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.ParameterTuple.html#mindspore.ParameterTuple)，用于保存多个`Parameter`，继承于元组`tuple`，提供克隆功能。
# 
# 如下示例提供`ParameterTuple`创建方法：

# In[10]:


import numpy as np
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer

# 创建
x = Parameter(default_input=Tensor(np.arange(2 * 3).reshape((2, 3))), name="x")
y = Parameter(default_input=initializer('ones', [1, 2, 3], mstype.float32), name='y')
z = Parameter(default_input=2.0, name='z')
params = ParameterTuple((x, y, z))

# 从params克隆并修改名称为"params_copy"
params_copy = params.clone("params_copy")

print(params)
print(params_copy)


# ## 网络参数初始化
# 
# MindSpore提供了多种网络参数初始化的方式，并在部分算子中封装了参数初始化的功能。本节以`Conv2d`算子为例，分别介绍使用`Initializer`子类，字符串和自定义`Tensor`等方式对网络中的参数进行初始化。
# 
# ### Initializer初始化
# 
# 使用`Initializer`对网络参数进行初始化，示例代码如下：

# In[11]:


import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import set_seed
from mindspore.common import initializer as init

set_seed(1)

input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
# 卷积层，输入通道为3，输出通道为64，卷积核大小为3*3，权重参数使用正态分布生成的随机数
net = nn.Conv2d(3, 64, 3, weight_init=init.Normal(0.2))
# 网络输出
output = net(input_data)


# ### 字符串初始化
# 
# 使用字符串对网络参数进行初始化，字符串的内容需要与`Initializer`的名称保持一致(字母不区分大小写)，使用字符串方式进行初始化将使用`Initializer`类中的默认参数，例如使用字符串`Normal`等同于使用`Initializer`的`Normal()`，示例如下：
# 

# In[12]:


import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import set_seed

set_seed(1)

input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init='Normal')
output = net(input_data)


# ### 张量初始化
# 
# 用户也可以通过自定义`Tensor`的方式，来对网络模型中算子的参数进行初始化，示例代码如下：

# In[13]:


import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype

init_data = Tensor(np.ones([64, 3, 3, 3]), dtype=mstype.float32)
input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))

net = nn.Conv2d(3, 64, 3, weight_init=init_data)
output = net(input_data)

