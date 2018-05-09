---
layout: post
title: Understanding of CapsNet
date: 2018-05-12 21:00:00.000000000 +09:00
tags: CapsNet 
---



[TOC]

## 1. Defects of CNNs

近些年CNN的巨大成功极大地促进了深度学习的发展，但是CNN的设计存在一个根本上的缺陷。以下图为例，考虑一张人脸，组成人脸的局部特征包括代表脸型的椭圆、两只眼睛、一个鼻子和一个嘴巴。CNN在浅层网络中学习检测边缘和颜色渐变等简单特征，而在深层中将这些简单特征组合成复杂特征，最后对特征组合并输出分类预测结果。在这一过程中，高层特征将底层特征进行组合是通过加权和和的方式，组成高层特征之间的底层特征之间并不存在**位姿**（平移和旋转）关系。即，CNN的内部数据表达没有考虑简单与复杂对象之间的重要空间层级关系。因此这两张图片在CNN看来是类似的，甚至都会被检测为人脸。另外，CNN通过最大池化来增加高层网络的“视野”，使得输入图像较大区域的高阶特征得以被检测，但是最大池化过程中损失了图像上有价值的信息。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/1.png)

Hinton等人通过对CNN设计缺陷的思考，提出大脑中物体的表示并不依赖于视角的思想，即从眼睛接收到的视觉信息中，大脑解析出我们周围世界的分层表示，并尝试匹配已学习到的模式和存储在大脑中的关系。因此Hinton放弃用标量作为神经元输出的设计，将检测到的特征的状态信息用向量表示，并提出了胶囊（capsule）网络。



## 2. What is capsule

在论文Dynamic Routing Between Capsules中，Capsule的定义如下：

> A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or an object part. We use the length of the activity vector to represent the probability that the entity exists and its orientation to represent the instantiation parameters. Active capsules at one level make predictions, via transformation matrices, for the instantiation parameters of higher-level capsules. When multiple predictions agree, a higher level capsule becomes active.

翻译过来是：

> “胶囊”是一组神经元，其活动向量表示特定实体类型（例如对象或对象部分）的实例化参数。我们使用活动向量的长度来表示实体存在的概率，方向表示实例化参数。在一个级别上的活性胶囊通过变换矩阵来预测更高级别的胶囊的实例化参数。当多个预测赞同时，一个更高级别的胶囊被激活。

暂时看不明白没关系。先记住几个概念：

1. 一个胶囊是**一组神经元**，并有其活动向量
2. 胶囊的活动向量的**长度**代表预测某一实体存在的**概率**，**方向**表示代表实体属性的**实例化参数**

在计算机图形学中，给定一组实例化参数（坐标，角度），加以渲染，我们就能得到右边的图形组合。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/2.png)

在逆图形学过程中，通过相反的过程，从图片中学习其组成对象及其实例化参数。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/3.png)

胶囊网络是一个类似逆图形的神经网络，由许多胶囊组成。如下图所示，包含50个胶囊，蓝色箭头表示三角形，黑色箭头表示矩形。箭头的方向和长度用来表示胶囊的活动向量。从图中可以看出来，在图中三角形和矩形存在的区域，相应胶囊的对应的活动向量的长度较大（表示预测三角形或矩形存在的概率较大），并且其方向与实体的位姿属性参数（此处为角度）对应。而在其他区域，胶囊活动限量的长度很小，表示该区域实体存在的概率很小，意味着在这些区域没有检测到任何东西。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/4.png)

现在回头想一想CNN和胶囊网络的区别：CNN没有编码特征之间的相对空间关系，最大池化丢失了有价值的信息。而在胶囊网络中，胶囊检测的特征信息均以向量的形式表示，向量长度代表特征被检测的概率，向量方向代表特征的状态，当特征在图像中位姿关系变化时，概率（向量长度）不变，但是方向发生变化，这就是Hinton提出的**活动等变性**：神经活动将随着物体在图像中的“外观流形上的移动”而改变。与此同时，检测概率保持恒定，这才是我们应该追求的那种不变性，而不是CNN提供的基于最大池化的不变性。如下图，稍微改变图像的角度，对应胶囊的向量也随之发生变化。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/5.png)

## 3. How does CapsNet work

**Squashing**

由于胶囊的长度表示一个概率，因此它不能大于1，论文中提出一个称为“**squashing**”的方法，保留向量的方向，但是将其长度压缩到0和1之间。

$$\mathbf v_j=\frac{\vert\vert \mathbf s_j\vert\vert^2}{1+\vert\vert  \mathbf s_j\vert\vert^2} \frac{ \mathbf s_j}{\vert\vert  \mathbf s_j\vert\vert}$$

其中$\mathbf v_j$是胶囊$j$的向量输出，$\mathbf s_j$是其总输入。squashing函数是对向量进行操作，纬度可能非常高导致无法进行可视化，可自行类比一维的图像。

**胶囊层级结构**

如下图所示，改变三角形和矩形的角度，我们可以将其组合成一所房子或一条船。在胶囊网络的逆图形学过程中，我们希望学习图片中的层级结构 ，即我们希望能识别船是船，而不是房子。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/6.png)

**胶囊网络工作过程**

一开始，我们使用若干层卷积层（CapsNet并不意味着放弃卷积）得到特征平面的数组表示，然后将其reshape成向量形式，接着采用squashing操作，将向量压缩到0-1之间。

接下来是胶囊网络最关键、最核心的操作：**前一层的每一个胶囊都试图预测后一层的胶囊的输出**。

如下图所示，假设第一层中有检测矩形和检测三角形的胶囊，我们分别称之为矩形胶囊和三角形胶囊，下一层只有两个胶囊：船胶囊和房胶囊。当矩形胶囊检测到图中矩形旋转了16度之后，它将会预测下一层的船倾斜了16度，同时它对房子的预测也是倾斜了16度。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/7.png)

而对于三角形胶囊，它将会预测房子是倒立的（根据三角形的角度），同时预测船倾斜了16度，这和矩形胶囊的预测是一致的。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/8.png)

结合上面的分析，我们可 以看出，三角形胶囊和矩形胶囊都非常赞同下一层输出船而不是房子，因为船能很好地解释它们的位姿（这里是角度）关系。因此，三角形和矩形都应将其输出只发送给船胶囊，使得船胶囊具有较大的激活值。这种机制被称为“**routing by agreement**”。这种机制有几个好处，首先，底层胶囊的输出只会发送给合适的高层胶囊，使得高层的这些胶囊获得更稳定的输入信号，能够更精确地确定高层特征的位姿关系。其次，routing by agreement机制给高层特征提供了一个可解释的层级关系，即我们识别出来的船是船，而不是房子。最后，routing by agreement机制有助于对重叠物体的检测和切分。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/9.png)

## 4. Dynamic routing between capsules

首先介绍网络的参数设置，如下图所示，$\mathbf W$表示变换矩阵，在一开始先对层底的输出向量$\mathbf u$进行矩阵乘法，得到变换后的向量$\hat{\mathbf u}$，即

$$\hat{\mathbf u}_{j\vert i}=\mathbf W_{ij}\mathbf u_i​$$ 

其中下标i和j表示上一层的第`i`个胶囊和下一层的第`j`个胶囊。$c_{ij}$表示标量权重，低层胶囊`i`改变权重$c_{ij}$，输出向量${\hat {\mathbf u}}_{j \vert i}$乘以该权重后，发送给高层胶囊`j`，作为高层胶囊的输入。由于$c_ {ij}$定义了传给每个高层胶囊`j`的输出的概率分布，因此对每个底层胶囊`i`而言，所有权重$c_ {ij}$的和为1。$\mathbf s_ {ij}$表示对高层胶囊`j`的总输入向量，有底层胶囊的预测向量和对应权重的加权和表示：

$$\mathbf s_j=\Sigma_ic_{ij}\hat{\mathbf u}_{j\vert i}​$$

最后，胶囊`j`的输出$\mathbf v_j$对$\mathbf s_j$进行squashing操作。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/11.png)

下面具体来看动态路由算法：

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/10.png)

算法第一行指明了输入：低层l中的所有胶囊及其输出$\hat{\mathbf u}$，以及路由迭代计数r。最后一行指明了输出，一个高层胶囊$\mathbf v_j$。

第2行的$b_{ij}$是一个临时变量，它的值会在迭代过程中更新，当整个算法运行完毕后，它的值计算一个softmax函数被保存到$c_{ij}$。在训练开始时，$b_{ij}$的值被初始化为零。

第3行表明路由迭代`r`次。

第4行计算向量$\mathbf c_{i}$的值，也就是低层胶囊i的所有权重。这一计算将应用到所有低层胶囊上。Softmax确保所有权重$c_{ij}$均为非负数，且其总和等于一。由于所有$b_{ij}$的值初始化为零，所以第一次迭代后，所有系数$c_{ij}$的值相等。例如，如果我们有3个低层胶囊和2个高层胶囊，那么所有$c_{ij}$将等于0.5。算法初始化时，所有$c_{ij}$均相等，这意味着不确定性达到最大值：低层胶囊不知道它们的输出最适合哪个高层胶囊。当然，随着这一进程的重复，这些均匀分布将发生改变。

第5行计算经前一步确定的路由系数$c_{ij}$加权后的输入向量的线性组合。从直觉上说，这意味着缩小输入向量并将它们相加，得到输出向量$\mathbf s_j$。这一步骤将应用到所有高层胶囊上。

在第6行中，来自前一步的向量将穿过squashing非线性函数，确保向量的方向被保留下来，而长度被限制在0-1。该步骤生成传给所有高层胶囊的输出向量$\mathbf v_j$。

第7行表示路由算法的权重更新策略，也是路由算法的本质所在。胶囊`j`的当前输出和从低层胶囊`i`处接收的输入的点积，加上旧权重，等于新权重。点积检测胶囊的输入和输出之间的相似性，相似度高的将会增大权重，反之则减小权重。

最后算法跳转到第3行重新开始这一流程，并重复`r`次。

## 5. CapsNet architecture

CapsNet由两部分组成：编码器和解码器。前3层是编码器，后3层是解码器：

**编码器**

第一层：卷积层

第二层：Primary主胶囊）层

第三层：DigitCaps（数字胶囊）层

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/13.png)

**解码器**

第四层：第一全连接层

第五层：第二全连接层

第六层：第三全连接层

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-05-10/14.png)

**Margin loss**

以手写数字分类问题为例，我们使用向量的长度代表实体的存在概率，因此我们希望当图片中存在数字时，对应的数字胶囊能够具有较大的向量长度。在论文中使用了称为margin loss的损失函数。

$$L_k=T_k max(0,m^+-\vert \vert \mathbf v_k\vert\vert)^2+\lambda(1-T_k)max(0,\vert \vert \mathbf v_k\vert \vert-m^-)^2$$

DigitCaps层的输出是10个16维向量，在训练时，对于每个训练样本，根据上述公式计算每个向量的损失值，然后将10个损失值相加得到最终损失。对于监督学习，每个训练样本都有正确的标签，在这种情况下，它将是一个10维one-hot向量，该向量由9个零和1个一（正确位置）组成。在损失函数公式中，正确的标签决定了$T_k$的值：如果正确的标签与特定DigitCap的数字对应，$T_k$为1，否则为0。

假设正确的标签是1，这意味着第一个DigitCap负责编码数字1的存在。这一DigitCap的损失函数的$T_k$为1，其余9个DigitCap的$T_k$为0。当$T_k$为1时，损失函数的第二项为零，损失函数的值通过第一项计算。在我们的例子中，为了计算第一个DigitCap的损失，我们从m+减去这一DigitCap的输出向量，其中，m+取固定值0.9。接着，我们保留所得值（仅当所得值大于零时）并取平方。否则，返回0。换句话说，当正确DigitCap预测正确标签的概率大于0.9时，损失函数为零，当概率小于0.9时，损失函数不为零。

对不匹配正确标签的DigitCap而言，$T_k$为零，因此将演算第二项。在这一情形下，DigitCap预测不正确标签的概率小于0.1时，损失函数为零，预测不正确标签的概率大于0.1时，损失函数不为零。

最后，公式包括了一个$\lambda$系数以确保训练中的数值稳定性（$\lambda$=0.5）。

## 6. Implementation of CapsNet

Github上有很多CapsNet的实现，推荐几个如下：

[Tensorflow implementation](https://github.com/naturomics/CapsNet-Tensorflow)

[Keras implementation](https://github.com/XifengGuo/CapsNet-Keras)

[Pytorch](https://github.com/adambielski/CapsNet-pytorch)

## 7. Comment

胶囊网络作为Hinton等人提出的一个新版本的神经网络，旨在解决CNN设计上的缺陷。胶囊网络保留了特征间的位姿信息，目前仅需较少的数据就能在MNIST等基准数据集上获得很高的准确度，其“routing by agreement”的策略也给重叠物体的检测和识别提供了一个可解释的方法。但是胶囊网络目前的缺点也很明显，首先，尚未明确在大数据集上的效果，其次，“routing by agreement”算法的内循环，使得网络的训练比较缓慢，最后，胶囊网络无法识别同一张图片上两个很接近的目标。总的来说，Hinton出品，其设计理念和价值都非常重大，但在其上仍需要进行大量的工作来不断改进。

## Declaration

This article is my personal understanding of CapsNet, There may be some error.  any criticism and suggestions are welcome. I mainly referenced as follow：

1. https://www.jqr.com/news/008838
2. https://www.youtube.com/watch?v=pPN8d0E3900  (highly recommended)

If there is any infringement, I will delete it immediately.

and you can find the slides [here](https://www.youtube.com/redirect?event=video_description&v=pPN8d0E3900&q=https%3A%2F%2Fwww.slideshare.net%2Faureliengeron%2Fintroduction-to-capsule-networks-capsnets&redir_token=v2mKhjk0uyX2XzejnAFRn0JRFsZ8MTUyNTkxNjU5MEAxNTI1ODMwMTkw).