---
layout: post
title: Batch Normalization
date: 2018-03-22 12:00:00.000000000 +09:00
tags: Machine learning  dropout  Deep learning
---

[TOC]

### Introduction

随机梯度下降(SGD) 是一个训练深度神经网络的有效方法。通常其通过优化如下loss来优化网络的参数$\Theta$:

$$\Theta=\mathop{argmax}\limits_{\Theta}\frac{1}{N} \mathop{l}(x_i,\Theta)$$

其中，$x_{1…N}$为训练集。在训练时，每次使用一个大小为m的mini-batch $x_{1…m}$，该mini-batch通过如下计算来近似loss的梯度：

$$\frac{1}{m} \frac{\partial l(x_i,\Theta)}{\partial \Theta}$$

使用mini-batch通常有以下几个好处：

1. 一个mini-batch的梯度可以近似看成整个数据集的梯度，尤其是size比较大时；
2. 并行的计算一个batch比计算m个单独的样本会更高效。

尽管SGD简单高效，但是，每一层的参数更新会导致上层的输入数据分布发生变化，随着网络层的加深，高层的输入分布变化会非常大，使得高层需要不断地重新适应底层的参数更新，这就是通常所说的covariate shift。这使得我们在炼丹时需要非常小心地设置学习率以及其他初始化参数。

另外，考虑一个网络计算如下loss：

$$l = F_2(F_1(u,\Theta_1),\Theta_2)$$

$F_1,F_2$为任意变换，$\Theta_1,\Theta_2$是最小化loss需要学习的网络参数。这时学习$\Theta_2$可以看作将输入$x=F_1(u,\Theta_1)$送进一个子网络：

$$l=F_2(x,\Theta_2)$$

例如我们用如下梯度下降策略更新网络权值：

$$\Theta_2 \leftarrow \Theta_2- \frac{\alpha}{m} \Sigma _{i=1}^{m} \frac{\partial{F_2(x_i,\Theta_2)}}{\partial{\Theta_2}}$$

其中batch size为$m$，学习率为$\alpha$. 此时网络的训练可以看成在训练一个单独的$F_2$，其输入是$x$。这样，随着网络的训练，$x$可能陷入一个固定的分布，$\Theta_2$不需要再进行调整来补偿输入$x$的分布的变化。

考虑一个使用Sigmoid激活函数的网络成${z = g(Wu+b)}$，${u}$是输入，${W}$是权值矩阵，$g(x)=\frac{1}{1+exp(-x)}$。随着${|x|}$增加，${g'(x)}$趋于0，这意味着，除了${|x|}$很小时，梯度很容易陷入非线性饱和区。这使得网络在深层很难收敛。

### Basic Concepts

#### Independent and Identically Distribution

独立同分布(i.i.d.) 是指数据样本**服从同一分布**并且**相互独立**。这并不是所有机器学习模型的前提假设条件，但是独立同分布的数据可以简化模型训练，并且预测能力更强。

#### Weighting

白化(Weighting)是数据预处理的一种手段，其目的是降低数据之间的冗余性，主要包括如下两个步骤：

1. 去除(或降低)数据之间的相关性——>去相关性——>**独立**
2. 使所有数据具有相同的均值和方差——>e.g. 均值0，方差1——>**同分布**

#### Internal Covariate Shift (ICS)

ICS是指网络节点输入的分布随着训练发生变化，因为每一层的参数更新都会导致上层的输入数据分布发生变化。[知乎回答](https://www.zhihu.com/question/38102762/answer/85238569)对此做了一个很有意思的解释：

> 大家都知道在统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如 transfer learning / domain adaptation 等。而 covariate shift 就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即：对所有![x\in \mathcal{X}](https://www.zhihu.com/equation?tex=x%5Cin+%5Cmathcal%7BX%7D),
>
> $$P_s(Y|X=x)=P_t(Y|X=x)$$
>
> 但是
>
> $$P_s(x) \not=P_t(X)$$
>
> 大家细想便会发现，的确，对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了covariate shift的定义。由于是对层间信号的分析，也即是“internal”的来由。

ICS带来的问题，[知乎回答](https://zhuanlan.zhihu.com/p/33173246)归纳得很好：

1. 上层参数需要不断适应新的输入数据分布的变化，降低学习速率；
2. 下层输入的变化可能趋向于过大或过下，落入饱和区，使得学习过早停止；
3. 每层的更新都影响到其他层，因此每层的参数更新策略需要尽可能的谨慎。

### General Framework of Normalization

以神经网络中一个神经元为例，其输入为：

$$\textbf{x} = (x_1,x_2,…,x_n)$$

输出为：

$$y=f(\textbf{x})$$

$\textbf{x}$ 的分布可能相差很大(因为ICS问题)，要解决独立同分布的问题，最好的的方法是对每一层的所有数据都进行白化操作。但是标准的白化操作代价很大，并且我们还希望白化操作是可微的，保证白化操作后可以通过反向传播来更新梯度。

通用的Normalization方法可以看成简化版的白化操作：

1. 在$\textbf{x}$ 输入之前先对其进行shift(平移)和scale(伸缩)变换，将$\textbf{x}$ 的分布规范化为在固定范围内的标准分布。

   $$\hat x = \frac{\textbf{x}-\mu }{\sigma}$$

​       其中$\mu$ 是**平移参数**，$\sigma$是**缩放参数**。这一步使得所有数据符合**均值为0，方差为1**的标准分布。

2. 但是这样变换之后的数据很可能降低网络原始的表达能力，因为第一步中会将几乎所有数据映射到激活函数的非饱和区(线性区)。因此进一步对数据进行re-shift(再平移)和re-scale(再缩放)变换，使数据重新获得非线性的表达能力：

   $$y=g\cdot\hat{x}+b$$

   其中$b$ 是**再平移参数****，$g$是**再缩放参数***。这一步使得所有数据的分布确定在**均值为$b$，方差为$g^2$**的区间。

因此总体的变换如下：

$$h = f\big(g\cdot \frac{\hat{x}-\mu}{\sigma}+b\big)$$

> 这里可以看出，Normalization只是将数据映射到一个确定的分布区间，并且未考虑去相关性的操作，因此距离标准的白化操作还很远。

### Batch Normalization

批规范化(batch normalization)顾名思义就是针对一个batch进行的normalization。如下图所示，计算一个mini-batch的均值和方差来估计神经元输入的均值和方差。$\beta$和$\gamma$是需要学习的再平移和再缩放参数。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2018-03-22/Figure_1.png)

在Batch Normalization中，将每一个 mini-batch 的统计量看成是对整体统计量的近似估计，或者说认为每一个 mini-batch 彼此之间，以及和整体数据，都应该是近似同分布的。分布差距较小的 mini-batch 可以看做是为规范化操作和模型训练引入了噪声，可以增加模型的鲁棒性；但如果每个 mini-batch的原始分布差别很大，那么不同 mini-batch 的数据将会进行不一样的数据变换，这就增加了模型训练的难度。因此，BN的**适用场景**是：每个 mini-batch 比较大，数据分布比较接近。在进行训练之前，要做好充分的 shuffle. 否则效果会差很多。

### Reference

1. [https://arxiv.org/pdf/1502.03167)](https://arxiv.org/pdf/1502.03167)
2. [https://zhuanlan.zhihu.com/p/33173246](https://zhuanlan.zhihu.com/p/33173246)







