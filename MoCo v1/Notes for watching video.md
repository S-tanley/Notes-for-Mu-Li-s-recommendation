## 对比学习

 无监督，有点像聚类算法，就是能自己分出类别，能知道图片谁和谁是一类，谁和谁不是一类。

但其实还是需要信息，需要知道谁跟谁是一类，谁跟谁不是一类。之所以它被认为是无监督，是因为我们可以使用一些办法去生成数据集，也就是说我们不需要手动标号，而是用我们生成的数据来训练模型。

比如：instant discrimination，每张图片自成一类，其他的否跟他不是一类。

对比学习的好处就是，想怎么比就怎么比，标准很简单，就是分出正负样本。

MoCo是第一个，全面的让无监督训练跟有监督训练相比，效果差不多甚至更好。

 MoCo用的是Momentum Contract
$$
y_t = m\cdot y_{t-1} + (1 - m)\cdot x_t
$$
加权移动平均

不想让当前时刻的输出完全依赖当前时刻的输入，所以有前一时刻的输出这一个项。

可以学到很好的特征，很好做迁移训练。

 构造动态字典

两点：

1. 首先就是用queue来抽特征，就是只有一部分被抽
2. 因为只有一部分别抽，所以每次都不一样，为了确保encoder similar，用momentum encoder，会取决于上一次的encoder，从而确保相似。

NEC loss

把类别变少了，为了loss算的快

其实就是cross entropy loss，K+1类的分类

方法核心，把字典看作队列，batch-size小，但是队列无所谓，计算开销可以很小

怎么更新key的编码器，不能改变的太快，要不然会降低一致性，动量encoder就是可以确保一致性。

这个其实讲过

为什么之前的方法会受限于这两点：end-to-end就是计算开太大，字典大小太大。

memory bank牺牲一致性，只有一个encoder for query。他会更新key，不同时刻的编码器得到的，缺乏一致性。 

trick: Shufflng BN

信息泄露

打乱送到GPU，再送回来

<img src="/Users/zhengbowen/Library/Mobile Documents/com~apple~CloudDocs/Paper library/Muli‘s recommended paper/MoCo v1/${img}/image-20241227153227507.png" alt="image-20241227153227507" style="zoom:25%;" />

A main goal of unsupervised learning is to *learn features that are transferrable*. 

 