---
layout: post
title: Pyplot Tutorial
date: 2017-12-13 12:30:00.000000000 +09:00
tags: python
---

## Pyplot简介

​	对matplotlib.pyplot的介绍，用官方的说法最贴切不过：Provides a MATLAB-like plotting framework。本文将主要介绍pyplot的基本使用。

​	[`matplotlib.pyplot`](http://matplotlib.org/api/pyplot_api.html#module-matplotlib.pyplot)是一系列让matplotlib使用起来和MATLAB相似的命令函数的集合。每一个`pyplot`函数将对一个图(figure)产生一些改动，例如创建绘图区域、绘图、对图增加label文字修饰等。

```python
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
```

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_1.png)

如上图所示，当给`plot()`仅提供一个列表或者数组是，默认是y轴的值，同时会自动产生x值，由于python的范围是从0开始，因此`y=[1,2,3,4]`时，`x=[0,1,2,3]`。

`plot(x-value,y-value,format-string...)`可以有任意多个参数，每3个参数为一组，分别是x,y和格式化字符串参数，其中格式化字符串参数为可选参数，提供对图上颜色、线条格式的定义。

```python
import matplotlib.pyplot as plt
# 'ro' = red circles
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])
plt.show()
```

![figure2](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_2.png)

`axis(xmin,xmax,ymin,ymax)`确定轴的范围，分别是x轴和y轴的最小最大值。

通常建议使用numpy数组，事实上，matplotlib会将所有数值序列转为numpy数组再进行处理。

```python
import numpy as np
import matplotlib.plt as plt
t = np.arange(0.,5.,0.2)
# red dashes,blue squares and green triangles
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()
```

![figure3](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_3.png)

## 控制线条属性

图的线条有许多属性可以设置，如宽度，线条风格（类型），抗锯齿等。可以使用如下方法控制线条属性：

* 使用关键字参数

```python
plt.plot(x,y.linewidth=2.0)
```

* 使用Line2D实例的 `setter`方法（`plot()`会返回一个Line2D实例的列表，如`line1,line2 = plot(x1,y1,x2,y2)`）

```python
line, = plt.plot(x,y,'-')
line.set_antialiased(False) # 关闭抗锯齿属性
```

* 使用`setp()`

```python
lines  =plt.plot(x1,y1,x2,y2)
# use keyword args
plt.setp(lines,color='r',linewidth=2.0)
# or MATLAB style string value pairs
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
```

线条的可设置属性可以通过将line或者lines作为参数传给`setp()`获得:

![figure4](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_4.png)

## Multiple figures and axes

和MATLAB一样，pyplot有当前`figure`和当前`axes`的概念，所有的绘图命令都是作用于当前`axes`的，`gca()`返回当前`axes`实例，`gcf()`返回当前`figure`实例。

```python
import numpy as np 
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0,5.0,0.1)
t2 = np.arange(0.0,5.0,0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1,f(t1),'bo',t2,f(t2),'k')

plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2),'r--')
plt.show()
```

![figure5](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_5.png)

`figure()`是可选的，默认会创建`figure（1）`，同样`subplot(111)`也会默认创建如果不显示说明。`figure(numrows,numcols,fignum)`的三个参数分别表示一个图（figure）上子图的行、列和当前子图的序号。因此一个图上共有`numrows*numcols`个子图，当`numrows*numcols<10`时，可以简写成`figure(211)`这样的形式。一个`figure`可以有任意多个`subplot`，也可以创建多个 `figure`。

```python
import matplotlib.pyplot as plt
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])

plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
```

> 注：使用`clf()`清空当前图，`cla()`清空当前轴，但是创建图之后，需要使用`close()`释放内存，仅仅关闭视图是没用的。

## 使用文本

`text()`函数可以在任意位置添加文本内容，`xlabel()`，`ylabel()`和`title()`用于在指定位置添加文本。

```python
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
```

![figure6](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_6.png)

所有的`text()`命令返回一个`matplotlib.text.Text`实例，可以通过关键字参数或者`setp()`为其定制属性：

```python
t = plt.xlabel('my data', fontsize=14, color='red')
```

不得不提的是，matplotlib竟然支持TeX，可以说是一个意外的惊喜。

```Tex
plt.title(r'$\sigma_i=15$')
```

常用的一个方法是为图上的某些特征添加标注，`annotate()`中xy关键字参数指明标注点位置，xytext关键字参数指明标注文本的位置，二者都是传入一个(x,y)的二元tuple。

```python
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.ylim(-2,2)
plt.show()
```

![figure7](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_7.png)

## 对数和其它非线性轴

`matplotlib.pyplot`不仅支持线性轴，也支持对数和对数尺度的轴，通常用于轴跨越多个数量级。

```python
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter  # useful for `logit` scale

# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure(1)

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()
```

![figure8](https://github.com/Pea-Shooter/Pea-Shooter.github.io/blob/master/images/blog/2017-12-13/Figure_8.png)