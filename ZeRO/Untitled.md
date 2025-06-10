# ZeRO

DeepSpeed就是基于Zero

主要是怎么将数据并行用于超大的训练集群上。

流水线并行，模型并行，数据并行，CPU-offloading，中间值重算

ADAM,维护一个momentum和一个variance。因为Nvidia用半精度算得快，造成大量的冗余。

内存碎片化，memory fragmentation。

Zero-dp，就是在做并行的时候，可能很多GPU都存了一些重复的信息，我只需要找一个GPU存就行了，其他的GPU要用的时候找这个GPU去要就好了。

状态信息，权重，梯度每个GPU都只维护原来的$1/N_d$.

ZeRO-R，带宽换空间，算好的东西放在不同的地方存。

怎么partition

$C_B$ constant size buffer, 就当机器够多的时候，每一层会被切很多片，每次发的数据就会很少。

内存整理，就判断一下那种一直会存的东西比如说梯度权重，放一块，另外就是随时会没的放一起。