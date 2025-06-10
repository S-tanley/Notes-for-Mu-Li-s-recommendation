# Generative Adversarial Nets

[TOC]

## Abstract

Proposing a new framework via an adversarial process. Two parts: one generative model *G*; and one discriminative model *D*. These two fight with each other, and in the end, we want G to generate new data that can fit training data perfectly but D cannot find these data are generated.

开门见山直接说我们提出了什么新模型，解释了一下模型，并说明我们想要的结果，提了一些优越性。

**Remark**: the title uses a simplification, we usually use networks but not nets, we better not do it, because it may cause some misunderstanding or unformal expression.



##  Introduction

DL有两种模型，discriminative models 和 generative models，discriminative models近些年进展很好，但是generative models却不行，因为就是计算似然函数的时候有很多困难。这篇文章其实就是在解决的generative models的问题，他们直接不去计算似然函数，而用别的方法得到更好的效果。

介绍了GAN，abstract也有，这里比较详细。

最后一段就是说他们的model很简单，很好计算。个人理解就是他们认为GAN除了效果很好以外的最大好处是很好计算，或者GAN之所以有很好的效果是因为他很好计算。

基本上是abstract的扩写，有一定故事性，比方说造假币那里。

##  Related work

之前的工作都是计算似然函数，但是计算会很难，所以有一些别的方法。Like Generative stochastic networks。这两个是有区别的。

之后说了一个相似的工作：VAEs。说了一下不一样的地方。

两个模型去竞争也有类似的想法，但是本文明确的指出来三点不同之处。

第四段说了“adversarial examples”，这个不是生成出来的，而是我们找的，跟正确样本很像的，可以糊弄分类器，去测试算法的稳定性。















## Adversarial nets















## Theoretical Results





### Global Optimality of $p_g = p_{data}$





### Convergence of Algorithm 1









## Experiments











## Advantages and disadvantages





## Conclusions and future work









## In The End

语言比较丰富，然后有一定故事性，和之前的不一样。







