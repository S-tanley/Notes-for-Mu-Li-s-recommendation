[TOC]

# Transformer

* 单头Attention最经典公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

+ 多头Attention（Transformer中常用）：


$$
\text{MultiHead}(Q,K,V) = \text{Concat}\left(\text{softmax}\left(\frac{QW_i^Q(KW_i^K)^T}{\sqrt{d_k}}\right)VW_i^V\right)_{i=1}^{h} W^O
$$








## Attention

Transformer 模型中的 **Attention 机制** 是其核心组成部分，主要用于捕捉输入序列中不同位置之间的依赖关系。Attention 的计算过程可以分为以下几个步骤：

---

### 1. 输入表示

假设我们有一个输入序列 $ X = [x_1, x_2, \dots, x_n] $，其中每个 $ x_i $ 是一个 $ d $-维向量（例如词嵌入）。输入序列可以表示为矩阵：

$$
X \in \mathbb{R}^{n \times d}
$$

其中 $ n $ 是序列长度，$ d $ 是每个向量的维度。

---

### 2. 线性变换

首先，通过线性变换将输入 $ X $ 映射到三个不同的空间：

- **Query (Q)**：用于查询其他位置的信息。
- **Key (K)**：用于被其他位置查询。
- **Value (V)**：包含实际的信息。

具体计算如下：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

其中：

- $ W_Q \in \mathbb{R}^{d \times d_k} $、$ W_K \in \mathbb{R}^{d \times d_k} $、$ W_V \in \mathbb{R}^{d \times d_v} $ 是可学习的权重矩阵。
- $ d_k $ 和 $ d_v $ 分别是 Query/Key 和 Value 的维度（通常 $ d_k = d_v = d $）。

---

### 3. 计算 Attention 分数

Attention 分数表示序列中每个位置对其他位置的重要性。通过计算 Query 和 Key 的点积来得到：

$$
\text{Attention Scores} = QK^T
$$

其中：

- $ Q \in \mathbb{R}^{n \times d_k} $，$ K \in \mathbb{R}^{n \times d_k} $，因此 $ QK^T \in \mathbb{R}^{n \times n} $。
- 每个元素 $ (QK^T)_{ij} $ 表示第 $ i $ 个位置的 Query 与第 $ j $ 个位置的 Key 的相似度。

---

### 4. 缩放和 Softmax

为了稳定梯度，通常会对 Attention 分数进行缩放（Scaled Dot-Product Attention）：

$$
\text{Scaled Attention Scores} = \frac{QK^T}{\sqrt{d_k}}
$$

然后通过 Softmax 函数将分数转换为概率分布：

$$
\text{Attention Weights} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

其中：

- Softmax 确保每个位置的权重和为 1。
- $ \sqrt{d_k} $ 是缩放因子，用于防止点积值过大导致梯度消失。

---

### 5. 加权求和

使用 Attention 权重对 Value 进行加权求和，得到每个位置的输出：

$$
\text{Output} = \text{Attention Weights} \cdot V
$$

其中：

- $ V \in \mathbb{R}^{n \times d_v} $，因此输出 $ \text{Output} \in \mathbb{R}^{n \times d_v} $。



---

### 6. 详细例子

假设我们有一个输入序列 $ X $ 包含 2 个词，每个词的嵌入维度 $ d = 4 $：

$$
X = \begin{bmatrix}
1 & 2 & 3 & 4 \\
4 & 3 & 2 & 1
\end{bmatrix}
$$

假设 $ W_Q $、$ W_K $、$ W_V $ 分别为：

$$
W_Q = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
W_K = \begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
W_V = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$

#### 1. 计算 $ Q $、$ K $、$ V $

$$
Q = XW_Q = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}, \quad
K = XW_K = \begin{bmatrix}
2 & 1 \\
3 & 4
\end{bmatrix}, \quad
V = XW_V = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
$$

#### 2. 计算 Attention 分数

$$
QK^T = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
\begin{bmatrix}
2 & 3 \\
1 & 4
\end{bmatrix}
= \begin{bmatrix}
4 & 11 \\
11 & 24
\end{bmatrix}
$$

#### 3. 缩放和 Softmax

假设 $ d_k = 2 $，缩放后：

$$
\frac{QK^T}{\sqrt{2}} = \begin{bmatrix}
2.828 & 7.778 \\
7.778 & 16.971
\end{bmatrix}
$$

Softmax 后：

$$
\text{Attention Weights} = \begin{bmatrix}
0.016 & 0.984 \\
0.000 & 1.000
\end{bmatrix}
$$

#### 4. 加权求和

$$
\text{Output} = \text{Attention Weights} \cdot V = \begin{bmatrix}
0.016 & 0.984 \\
0.000 & 1.000
\end{bmatrix}
\begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
= \begin{bmatrix}
3.952 & 2.984 \\
4.000 & 3.000
\end{bmatrix}
$$

---

### 7. 总结

- Attention 的核心是计算 Query、Key 和 Value 之间的关系。
- 通过点积、缩放、Softmax 和加权求和，得到每个位置的输出。
- 多头 Attention 可以捕捉不同子空间的信息。



## Multi-Head Attention

Transformer 模型中的 **Multi-Head Attention（多头注意力机制）** 是其核心组成部分之一。它通过并行计算多个注意力头（Attention Heads），捕捉输入序列中不同子空间的信息，从而增强模型的表达能力。以下是多头注意力机制的具体实现步骤和详细例子。

---

### 1. 多头注意力的核心思想

- **单头注意力**：计算一组 Query、Key 和 Value，得到一个注意力输出。
- **多头注意力**：将 Query、Key 和 Value 拆分为多个头，每个头独立计算注意力，最后将多个头的输出拼接起来。

---

### 2. 多头注意力的实现步骤

#### 1. 输入表示

假设输入序列 $ X $ 是一个矩阵：

$$
X \in \mathbb{R}^{n \times d}
$$

其中：
- $ n $ 是序列长度。
- $ d $ 是每个向量的维度。

#### 2. 线性变换

将输入 $ X $ 通过线性变换映射到 Query、Key 和 Value：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

其中：
- $ W_Q \in \mathbb{R}^{d \times d_k} $、$ W_K \in \mathbb{R}^{d \times d_k} $、$ W_V \in \mathbb{R}^{d \times d_v} $ 是可学习的权重矩阵。
- $ d_k $ 和 $ d_v $ 分别是 Query/Key 和 Value 的维度（通常 $ d_k = d_v = d $）。

#### 3. 拆分多头

将 $ Q $、$ K $、$ V $ 拆分为 $ h $ 个头（例如 $ h = 8 $）。假设 $ d_k = d_v = d $，则每个头的维度为 $ d_k' = d_k / h $，$ d_v' = d_v / h $。

拆分后的 $ Q $、$ K $、$ V $ 分别为：

$$
Q_i = QW_{Q_i}, \quad K_i = KW_{K_i}, \quad V_i = VW_{V_i}
$$

其中：
- $ W_{Q_i} \in \mathbb{R}^{d_k \times d_k'} $、$ W_{K_i} \in \mathbb{R}^{d_k \times d_k'} $、$ W_{V_i} \in \mathbb{R}^{d_v \times d_v'} $ 是每个头的可学习权重矩阵。
- 每个头的输出维度为 $ \mathbb{R}^{n \times d_k'} $ 和 $ \mathbb{R}^{n \times d_v'} $。

#### 4. 计算每个头的注意力

对每个头 $ i $，计算注意力输出：

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$

其中：
- $ \text{Attention} $ 是标准的缩放点积注意力机制。

#### 5. 拼接多头输出

将多个头的输出拼接起来：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W_O
$$

其中：
- $ W_O \in \mathbb{R}^{h d_v' \times d} $ 是输出权重矩阵。
- 最终输出的维度为 $ \mathbb{R}^{n \times d} $。

---

### 3. 详细例子

假设：
- 输入序列 $ X $ 包含 2 个词，每个词的嵌入维度 $ d = 4 $。
- 多头注意力的头数 $ h = 2 $，因此每个头的维度 $ d_k' = d_v' = 2 $。

#### 1. 输入表示

输入序列 $ X $：

$$
X = \begin{bmatrix}
1 & 2 & 3 & 4 \\
4 & 3 & 2 & 1
\end{bmatrix}
$$

#### 2. 线性变换

假设 $ W_Q $、$ W_K $、$ W_V $ 分别为：

$$
W_Q = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
W_K = \begin{bmatrix}
0 & 1 \\
1 & 0 \\
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
W_V = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$

计算 $ Q $、$ K $、$ V $：

$$
Q = XW_Q = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}, \quad
K = XW_K = \begin{bmatrix}
2 & 1 \\
3 & 4
\end{bmatrix}, \quad
V = XW_V = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
$$

#### 3. 拆分多头

假设每个头的权重矩阵为：

$$
W_{Q_1} = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
W_{K_1} = \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
W_{V_1} = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

$$
W_{Q_2} = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
W_{K_2} = \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
W_{V_2} = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

计算每个头的 $ Q_i $、$ K_i $、$ V_i $：

$$
Q_1 = QW_{Q_1} = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}, \quad
K_1 = KW_{K_1} = \begin{bmatrix}
2 & 1 \\
3 & 4
\end{bmatrix}, \quad
V_1 = VW_{V_1} = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
$$

$$
Q_2 = QW_{Q_2} = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}, \quad
K_2 = KW_{K_2} = \begin{bmatrix}
2 & 1 \\
3 & 4
\end{bmatrix}, \quad
V_2 = VW_{V_2} = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
$$

#### 4. 计算每个头的注意力

对每个头 $ i $，计算注意力输出：

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$

以第一个头为例：

$$
\text{Attention Scores}_1 = Q_1K_1^T = \begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
\begin{bmatrix}
2 & 3 \\
1 & 4
\end{bmatrix}
= \begin{bmatrix}
4 & 11 \\
11 & 24
\end{bmatrix}
$$

$$
\text{Scaled Attention Scores}_1 = \frac{\text{Attention Scores}_1}{\sqrt{2}} = \begin{bmatrix}
2.828 & 7.778 \\
7.778 & 16.971
\end{bmatrix}
$$

$$
\text{Attention Weights}_1 = \text{Softmax}\left(\frac{Q_1K_1^T}{\sqrt{2}}\right) = \begin{bmatrix}
0.016 & 0.984 \\
0.000 & 1.000
\end{bmatrix}
$$

$$
\text{head}_1 = \text{Attention Weights}_1 \cdot V_1 = \begin{bmatrix}
0.016 & 0.984 \\
0.000 & 1.000
\end{bmatrix}
\begin{bmatrix}
1 & 2 \\
4 & 3
\end{bmatrix}
= \begin{bmatrix}
3.952 & 2.984 \\
4.000 & 3.000
\end{bmatrix}
$$

同理，计算第二个头的输出 $ \text{head}_2 $。

#### 5. 拼接多头输出

将两个头的输出拼接起来：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2)W_O
$$

假设 $ W_O $ 为单位矩阵，则最终输出为：

$$
\text{MultiHead}(Q, K, V) = \begin{bmatrix}
3.952 & 2.984 & 3.952 & 2.984 \\
4.000 & 3.000 & 4.000 & 3.000
\end{bmatrix}
$$

---

### 4. 总结

- 多头注意力通过并行计算多个注意力头，捕捉输入序列中不同子空间的信息。
- 每个头独立计算注意力，最后将多个头的输出拼接起来，并通过线性变换得到最终输出。



## Self-Attention & Cross-Attention

### 一、Self-Attention 是什么？

Self-Attention（自注意力）是指：**Query、Key、Value 都来自同一个输入序列**，用于捕捉序列内部 token 之间的关系。

#### 结构公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：

- $Q = XW^Q$，$K = XW^K$，$V = XW^V$，X 是输入序列（n 个 token）
- 所以 Q/K/V 来自同一个 X，但经过不同的线性变换（$W^Q, W^K, W^V$ 是可学习参数矩阵）

#### 特点：

- 输入 = 输出长度
- 无信息遮挡（可看到前后文）
- 多用于表征学习、序列理解

#### 使用场景：

- Encoder 中（如 BERT）
- Decoder 中（如 GPT, 作为 Masked Self-Attention）

---

### 二、Cross-Attention 是什么？

Cross-Attention 是指：**Query 来自 Decoder，Key 和 Value 来自 Encoder 的输出**，用于捕捉“源-目标”之间的对齐关系。

#### 结构公式：

$$
\text{Attention}(Q^{dec}, K^{enc}, V^{enc})
$$

其中：

- $Q = X^{dec}W^Q，K = X^{enc}W^K，V = X^{enc}W^V$
- $X^{dec}$ 是 Decoder 的当前输入，$X^{enc}$ 是 Encoder 的输出
- 同样需要通过三个独立的 projection 矩阵获得 Q/K/V，不能直接使用原向量

#### 特点：

- Q ≠ K/V，跨序列计算注意力
- Decoder 用它来访问 Encoder 的上下文信息
- 在 Encoder-Decoder 架构中使用

#### 使用场景：

- Decoder 模块中（T5, BART）
- 多模态模型中（图文、音文对齐）

---

### 三、对比总结表

| 对比项         | Self-Attention                 | Cross-Attention     |
| -------------- | ------------------------------ | ------------------- |
| Q              | 来自当前序列                   | 来自 Decoder        |
| K/V            | 来自当前序列                   | 来自 Encoder        |
| 是否乘参数矩阵 | 是（W^Q, W^K, W^V）            | 是（W^Q, W^K, W^V） |
| 应用位置       | Encoder / Decoder              | Decoder             |
| 是否跨序列     | 否                             | 是                  |
| 是否有 masking | 有（Decoder 中）/无（Encoder） | 无                  |

---

### 四、在 Decoder 中的顺序结构

每一层 Decoder 都包含如下结构：

1. Masked Self-Attention（只看前文）
2. Cross-Attention（看 Encoder 输出）
3. Feed-Forward Network

其中 Cross-Attention 的 K/V 来自 Encoder 最后一层的输出，是固定的，不随 Decoder 层数变化。

---

### 五、小结

- 无论是 Self-Attention 还是 Cross-Attention，Q、K、V 都是通过线性变换得到的，即使它们来自同一个或不同的序列。
- Cross-Attention 的关键在于：Q 来自 Decoder，K/V 来自 Encoder，但依然要分别乘各自的可学习参数矩阵（W^Q, W^K, W^V）。
- 这种结构确保了模型能灵活、有效地实现信息对齐与融合。



## Encoder & Decoder

### 一、Encoder Block 结构

用于提取输入序列的上下文信息（双向表示）。

#### 每层结构：
1. **Self-Attention**（无 mask，允许双向关注）
2. **Feed-Forward Network (FFN)**
3. **Add & LayerNorm**（残差连接 + 归一化）

#### 特点：
- 可并行处理整个序列
- 所有 token 能看到全局
- 用于理解型任务（分类、检索）

---

### 二、Decoder Block 结构

用于生成序列，每一步基于之前已生成的内容。

#### 每层结构：
1. **Masked Self-Attention**（遮挡未来 token）
2. **Cross-Attention**（对 Encoder 输出做 attention）
3. **Feed-Forward Network (FFN)**
4. **Add & LayerNorm**（残差连接 + 归一化）

#### 特点：
- 自回归生成（逐 token）
- 可访问 Encoder 编码信息
- 用于生成型任务（翻译、对话）

---

### 三、三类模型架构对比

| 架构类型            | 结构组成           | 应用场景               | 代表模型          |
| ------------------- | ------------------ | ---------------------- | ----------------- |
| **Encoder-only**    | 只用 Encoder Stack | 分类、理解、检索等     | BERT、RoBERTa     |
| **Decoder-only**    | 只用 Decoder Stack | 文本生成、续写、补全   | GPT、LLaMA        |
| **Encoder-Decoder** | 编码 + 解码两部分  | 翻译、摘要、多模态生成 | T5、BART、Flan-T5 |

#### 特别说明：
- Encoder-only 中 Self-Attention 是双向的；
- Decoder-only 中 Self-Attention 是 masked 的；
- Encoder-Decoder 架构中，Decoder 每层都包含 Cross-Attention。

---

### 四、应用建议

| 任务类型       | 建议架构        |
| -------------- | --------------- |
| 情感分类、NER  | Encoder-only    |
| 对话、语言建模 | Decoder-only    |
| 翻译、摘要     | Encoder-Decoder |

