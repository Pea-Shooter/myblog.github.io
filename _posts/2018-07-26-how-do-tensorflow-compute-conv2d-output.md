---
layout: post
title: How do tensorflow compute conv2d output? 
date: 2018-07-26 10:00:00.000000000 +09:00
tags: tensorflow 
---

[TOC]

## 1. Methods of padding

* "VALID"=without padding

  "VALID" only ever drops the right-most columns (or bottom-most rows)

* "SAME"=with zero padding

  "SAME" tries to pad evenly left and right.



## 2. How do they compute the output?

* For the `SAME` padding, the output height and width are computed as

  ```python
  out_height = ceil(float(in_height) / float(strides[1]))
  out_width  = ceil(float(in_width) / float(strides[2]))
  ```

  The total padding applied along the height and width is computed as:

  ```python
  if (in_height % strides[1] == 0):
    pad_along_height = max(filter_height - strides[1], 0)
  else:
    pad_along_height = max(filter_height - (in_height % strides[1]), 0)
  if (in_width % strides[2] == 0):
    pad_along_width = max(filter_width - strides[2], 0)
  else:
    pad_along_width = max(filter_width - (in_width % strides[2]), 0)
  ```

  Finally, the padding on the top, bottom, left and right are:

  ```python
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  ```


* For the` VALID` padding, the output height and width are computed as 

  ```python
  out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
  out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
  ```



## 3. Parameter interpretation of conv2d

```p
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```

* input : A 4-D `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`. The dimension order is interpreted according to the value of `data_format`, default `[batch_num, in_height, in_width, in_channels]`.
* filter : A 4-D `Tensor`. Must have the same type as `input`. `[filter_height, filter_width, in_channels, out_channels]`
* strides : A list of `ints`. 1-D tensor of length 4. The stride of the sliding window for each dimension of `input`.
* data_format : An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`. Specify the data format of the input and output data. With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels]. Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].

The `in_channels of filter` = `in_channels of input` , `out_channels of filter` = `num of filters`. By default, strides[0]=strides[3]=1.



## 4. Example

If padding='VALID':

```python
import tensorflow as tf

input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,3]))
op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = (sess.run(op))
    print(res.shape)
```

the output is `(1, 3, 3, 3)`.

If padding='SAME':

```python
import tensorflow as tf

input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,3]))
op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = (sess.run(op))
    print(res.shape)
```

the output is `(1, 5, 5, 3)`.