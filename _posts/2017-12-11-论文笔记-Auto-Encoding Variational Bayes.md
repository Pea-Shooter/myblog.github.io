---
layout: post
title: Auto-Encoding Variational Bayes
date: 2017-12-11 23:00:00.000000000 +09:00
tags: machine learning,generative model
---

## Paper Info

Full name:**Auto-Encoding Variational Bayes**

Authors:Diederik P Kingma,Max Welling

[Link](https://arxiv.org/abs/1312.6114)

## Abstract

​	针对“当存在不确定后验分布的连续的潜在变量（continuous latent variables）和大数据集时，如何在定向概率模型上进行有效的推导和学习？”这一问题，作者提出了一种随机变分推导和学习算法，在一些温和的不同条件下，可以扩展到大数据集，甚至可以在棘手的情况下工作。本文的贡献有：（1）证明了变分下界的再参数化得到了一个较低的边界估计量，可以直接用标准随机梯度法进行优化；（2）证明了对于拥有连续潜在变量的独立同分布（i.i.d）数据集，后验推断（posterior inference）可以通过将一个近似推断模型与被提出的下界估计器进行匹配，从而使其更有效。

## Introduction

​	变分贝叶斯（VB）方法涉及到对难以处理的后验的近似的优化。不幸的是，一般的均值-场（mean-field）法需要的是关于后验的期望的解析解，通常这是难以处理的。本文展示了变分下界的再参数化如何产生一个简单的可微分的无偏估计下界，这种随机梯度变分贝叶斯（Stochastic Gradient Variational Bayes,SGVB）估计量可以用于几乎任何具有连续潜在变量和参数的模型的有效近似后验推断，并且可以直接使用标准的随机梯度上升技术进行优化。

​	对于独立同分布数据集和连续潜在变量样本的情况，本文提出了Auto Encoding VB（AEVB）算法，通过AEVB算法我们进行推断和学习，尤其是有效利用SGVB估计优化识别模型（recognition model），该模型允许我们通过简单的祖先采样（ancestral modeling）进行非常有效的近似推断，从而使我们在不需要昂贵的迭代推理方式（如MCMC）就可以有效的学习模型参数。学到的近似后验推断模型可以用于许多任务，例如识别（recognition）、去燥（denoising）、表达（representation）和可视化（visualization）。当神经网络采用这种模型时，就产生了VAE（Variational auto-encoder）。

## Method

**问题场景**

假设有N个独立同分布的连续或者离散的数据集样本 $X=\{x^{(i)}\}_{i=1}^{N}$ ，假设数据由某些随机过程产生并且包含一个未观测到的连续潜在变量。这个过程包含两个步骤

(1) $z^{i}$ 由先验分布$p_{\theta^*}z$产生

(2) $x^i$由条件分布$p_{\theta^*} (x\vert z)$

假设先验 $p_{\theta^{\star}}z​$ 和极大似然 $p_{\theta^{\star}}(x\vert z)​$ 来自分布 $p_\theta z​$ 和 $p_{\theta}(x\vert z)​$ 的参数簇，但是这个过程我们是无法观测的，真实参数 $\theta^*​$ 和隐变量 $z^{(i)}​$ 都是未知的。这里不对边缘或者后验概率分布做一般化假设，而是提出一种针对如下两个问题通用的算法:

(1)无法计算有非常棘手的真实后验密度$p_{\theta}(z\vert x)=p_{\theta}(x\vert z)p_{\theta}(z)/p_{\theta}(x)$的边缘分布的极大似然估计积分$\int p_{\theta}(z)p_{\theta}(x\vert z)dz$；

(2)对于一个大数据集，批量优化代价很大，minibatch或者单个数据点采样又非常慢。

**解决方案**

文章针对上述场景提出了变分下界、SGVB估计器、AEVB算法和再参数化trick，然后作为example提出了Variational Auto-Encoder（VAE），具体的推导与证明详见文章，由于比较繁琐与复杂，在此不做描述。

下面我们来具体看看VAE：

(1)Encoder：引入隐变量z，利用编码网络拟合参数化的后验 $q_{\phi}(z\vert x)$ ,输出为z的条件分布:

$$q_{\phi}(z\vert x)=N(z;\mu_{z}(x),\sigma_{z}(x))$$

其中$\mu_z(x)$,$\sigma_{z}(x)$是Encoder的输出。并且此处假设z的先验分布是N(0,1)。

![figure1](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2017-12-11/encoder.jpeg)

(2) Decoder：已知隐变量z，计算样本x的条件似然概率$p_{\theta}(x\vert z)$，$\theta$是参数。

![figure2](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2017-12-11/figure1.png)

(3) 整个网络最终的优化目标是确保Encoder中的后验概率密度函数逼近Decoder中的后验概率密度函数，目标函数为：

$$\mathcal{L}(\theta,\phi,x)=-D_{KL}(q_{\phi}(z\vert x)||p_{\theta}(z))+\mathbb{E}_{q_{\phi}(z\mid x)}[log\,p_{\theta}(x\vert z)]$$

网络结构图：

![figure3](https://github.com/Pea-Shooter/Pea-Shooter.github.io/raw/master/images/blog/2017-12-11/VAE.png)

> 注：网络结构图[来源](http://zhouchang.info/blog/2016-04-11/VAE.html)

针对VAE的推导与实现，推荐一篇[博客](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/)