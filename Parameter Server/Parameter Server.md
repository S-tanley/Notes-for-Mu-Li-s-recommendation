# Parameters Server

OSDI，系统领域的顶会。

在小地方去开

Big Data，学术界和工业界对big data有误解。

分布式

实现一个性能很好的分布式算法很难

计算复杂度很高，数据通讯量很大

大量机器同时用的时候，**容灾**很重要。训练量一大就很容易挂掉

数据做通讯的时候可以一段一段发，然后就是容灾。

![CleanShot 2025-01-31 at 18.01.47](/Users/zhengbowen/Library/Mobile Documents/com~apple~CloudDocs/Paper library/Muli‘s recommended paper/Parameter Server/${img}/CleanShot 2025-01-31 at 18.01.47.png)

（Key，Value） Vector

Range Push and Pull，一发发一段

vector clock，维护每个节点的异步性

只需要记录每一段的时间

一致性哈希环，容灾

