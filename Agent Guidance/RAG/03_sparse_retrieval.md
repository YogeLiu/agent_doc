# 稀疏检索（Sparse Retrieval）

## 概述

稀疏检索方法是基于关键词匹配和统计度量的基础信息检索算法。与神经网络产生的密集向量不同，稀疏向量主要包含零值，只有少数非零元素代表重要的词项。

**稀疏检索的主要优势：**

- 高可解释性 - 可以清楚看到哪些词项匹配
- 更好的域外泛化能力
- 大规模搜索的计算效率高
- 无需昂贵的神经模型训练

## BM25 算法

### 什么是 BM25？

Best Matching 25（BM25）是一种概率排序算法，改进了传统的 TF-IDF。它被广泛应用于搜索引擎，根据文档与查询的相关性对文档进行排序。

**相比 TF-IDF 的主要改进：**

1. **词频饱和度** - 解决了重复词项的边际收益递减问题
2. **文档长度归一化** - 防止对长文档的偏向
3. **可调参数** - 适应不同语料库的特征

### 先理解 TF-IDF

在深入了解 BM25 之前，理解 TF-IDF（词频-逆文档频率）很重要：

**词频（Term Frequency, TF）：**

$$\text{TF} = \frac{\text{词项在文档中出现的次数}}{\text{文档中的总词数}}$$

**逆文档频率（Inverse Document Frequency, IDF）：**

$$\text{IDF} = \log\left(\frac{\text{总文档数}}{\text{包含该词项的文档数}}\right)$$

**最终 TF-IDF 得分：**

$$\text{TF-IDF} = \text{TF} \times \text{IDF}$$

**TF-IDF 的局限性：**

- 词频与相关性之间是线性关系（未考虑饱和效应）
- 未考虑文档长度
- 无法很好地处理关键词堆砌问题

## 从 TF-IDF 到 BM25 的演化过程

### 缺陷 1：词频线性增长问题

**TF-IDF 的问题：**

在传统 TF-IDF 中，词频部分是线性增长的：

$$\text{TF} = \frac{f(q, D)}{|D|}$$

其中：

- $f(q, D)$ = 查询词项 $q$ 在文档 $D$ 中出现的次数
- $|D|$ = 文档 $D$ 的总词数

这意味着：

- 词项出现 2 次 → 4 次的影响 = 词项出现 200 次 → 202 次的影响
- 但实际上，当词项已经出现很多次时，额外的出现对相关性的贡献应该递减

**具体例子：**

假设我们搜索"兔子"：

- 文档 A：1000 词，"兔子"出现 10 次 → TF = 10/1000 = 0.01
- 文档 B：1000 词，"兔子"出现 100 次 → TF = 100/1000 = 0.1
- 文档 C：1000 词，"兔子"出现 400 次 → TF = 400/1000 = 0.4

根据 TF-IDF，文档 C 的相关性是文档 B 的 4 倍。但实际上，"兔子"出现 100 次已经足以说明文档与兔子相关，出现 400 次并不意味着相关性是 4 倍。

**改进步骤 1：引入词频饱和机制**

BM25 引入饱和函数来控制词频的边际收益递减：

$$\text{TF}_{\text{saturated}} = \frac{f(q, D)}{f(q, D) + k}$$

其中：

- $f(q, D)$ = 查询词项 $q$ 在文档 $D$ 中出现的次数
- $k$ = 饱和参数，控制词频饱和的速度（通常为 1.2）

**饱和效果分析：**

当 $k = 1.2$ 时：

- $f = 1$：$\frac{1}{1+1.2} = 0.45$
- $f = 2$：$\frac{2}{2+1.2} = 0.63$ （增加 0.18）
- $f = 4$：$\frac{4}{4+1.2} = 0.77$ （增加 0.14）
- $f = 10$：$\frac{10}{10+1.2} = 0.89$ （增加 0.12）
- $f = 100$：$\frac{100}{100+1.2} = 0.988$
- $f = 200$：$\frac{200}{200+1.2} = 0.994$ （增加 0.006）

可以看到，随着词频增加，额外出现的贡献越来越小，最终趋近于 1（饱和）。

### 缺陷 2：未考虑文档长度

**TF-IDF 的问题：**

TF-IDF 没有考虑文档长度对相关性的影响。

**具体例子：**

假设我们搜索"旅行"：

- 文档 A：10 词，"旅行"出现 1 次 → TF = 1/10 = 0.1
- 文档 B：1000 词，"旅行"出现 10 次 → TF = 10/1000 = 0.01

按照 TF-IDF，文档 A 的得分更高。但如果我们从另一个角度看：

- 文档 A：10 个词中有 1 个是"旅行"（10% 的内容）
- 文档 B：1000 个词中有 10 个是"旅行"（1% 的内容）

直觉上，文档 A 更可能专注讨论"旅行"，应该给予更高的权重。

**改进步骤 2：引入文档长度归一化**

BM25 引入文档长度归一化因子：

$$\text{TF}_{\text{normalized}} = \frac{f(q, D)}{f(q, D) + k \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

其中：

- $|D|$ = 当前文档长度
- $\text{avgdl}$ = 平均文档长度
- $b$ = 长度归一化参数（0 到 1 之间，通常为 0.75）

**归一化效果分析：**

令 $k = 1.2$，$b = 0.75$，$\text{avgdl} = 500$：

对于短文档（$|D| = 100$）：
$$k \cdot \left(1 - 0.75 + 0.75 \cdot \frac{100}{500}\right) = 1.2 \cdot (0.25 + 0.15) = 0.48$$

对于平均长度文档（$|D| = 500$）：
$$k \cdot \left(1 - 0.75 + 0.75 \cdot \frac{500}{500}\right) = 1.2 \cdot (0.25 + 0.75) = 1.2$$

对于长文档（$|D| = 1000$）：
$$k \cdot \left(1 - 0.75 + 0.75 \cdot \frac{1000}{500}\right) = 1.2 \cdot (0.25 + 1.5) = 2.1$$

可以看到：

- 短文档的分母更小 → 更容易达到饱和（更快获得高分）
- 长文档的分母更大 → 需要更多词频才能达到同样的相关性得分

**参数 $b$ 的作用：**

- $b = 0$：完全忽略文档长度，公式变为 $k \cdot 1 = k$
- $b = 1$：完全考虑文档长度，公式变为 $k \cdot \frac{|D|}{\text{avgdl}}$
- $b = 0.75$：折衷方案，部分考虑文档长度

### 缺陷 3：IDF 可能产生负值

**TF-IDF 的 IDF 问题：**

传统 TF-IDF 的 IDF 公式：

$$\text{IDF} = \log\left(\frac{N}{\text{DF}(q)}\right)$$

当词项出现在超过一半的文档中时（$\text{DF}(q) > N/2$），IDF 会变小，但不会变成负数（因为 $\text{DF}(q) \leq N$）。

然而，在某些改进的 TF-IDF 变体中，使用了更精细的 IDF 公式，可能导致负值问题。

**改进步骤 3：改进 IDF 公式**

BM25 使用更稳健的 IDF（经验所得）：

$$\text{IDF}(q) = \log\left(\frac{N - \text{DF}(q) + 0.5}{\text{DF}(q) + 0.5} + 1\right)$$

**IDF 公式的优点：**

1. **加入 0.5 平滑**：避免除零错误，处理极端情况
2. **加 1 防止负值**：确保 IDF 始终为正数
3. **更好的数学特性**：来自概率检索模型的理论推导

**IDF 值示例：**

假设语料库有 $N = 1000$ 个文档：

- 罕见词（$\text{DF} = 1$）：
  $$\text{IDF} = \log\left(\frac{1000-1+0.5}{1+0.5} + 1\right) = \log(667.67) \approx 6.5$$

- 常见词（$\text{DF} = 100$）：
  $$\text{IDF} = \log\left(\frac{1000-100+0.5}{100+0.5} + 1\right) = \log(9.96) \approx 2.3$$

- 非常常见词（$\text{DF} = 500$）：
  $$\text{IDF} = \log\left(\frac{1000-500+0.5}{500+0.5} + 1\right) = \log(2.0) \approx 0.69$$

- 出现在几乎所有文档（$\text{DF} = 900$）：
  $$\text{IDF} = \log\left(\frac{1000-900+0.5}{900+0.5} + 1\right) = \log(1.11) \approx 0.11$$

可以看到，即使词项出现在 90% 的文档中，IDF 仍然是正数，只是非常小。

### 最终的 BM25 公式

结合以上所有改进，我们得到完整的 BM25 公式：

$$\text{BM25}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k+1)}{f(q_i, D) + k \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

其中：

- $D$ = 文档
- $Q$ = 查询
- $q_i$ = 查询词项 i
- $f(q_i, D)$ = 词项 $q_i$ 在文档 $D$ 中的频率
- $|D|$ = 文档 $D$ 的长度
- $\text{avgdl}$ = 语料库中文档的平均长度
- $k$ = 词频饱和参数（通常为 1.2）
- $b$ = 文档长度归一化参数（通常为 0.75）
- $\text{IDF}(q_i) = \log\left(\frac{N - \text{DF}(q_i) + 0.5}{\text{DF}(q_i) + 0.5} + 1\right)$

**公式解读：**

1. **分子 $f(q_i, D) \cdot (k+1)$**：词频的基础权重，$(k+1)$ 是归一化因子
2. **分母饱和部分 $f(q_i, D) + k \cdot (...)$**：实现词频饱和和长度归一化
3. **IDF 权重**：惩罚常见词，奖励罕见词
4. **求和 $\sum$**：对查询中的所有词项累加得分

### 演化总结

| 问题           | TF-IDF               | BM25 解决方案                             |
| -------------- | -------------------- | ----------------------------------------- |
| 词频线性增长   | $f/\|D\|$            | $\frac{f}{f+k}$ 实现饱和                  |
| 忽略文档长度   | 无长度考虑           | 引入 $b \cdot \frac{\|D\|}{\text{avgdl}}$ |
| IDF 可能不稳定 | $\log(N/\text{DF})$  | 加入平滑项和 +1 防止负值                  |
| 无可调参数     | 固定公式             | $k$ 和 $b$ 可根据语料库调整               |
| 对长文档的偏向 | 长文档更容易得高分   | 通过 $b$ 参数平衡长短文档                 |
| 关键词堆砌问题 | 堆砌词频线性增加得分 | 饱和机制限制堆砌效果                      |

### BM25 组成部分详解

#### 1. 词频饱和度（k 参数）

k 参数控制词项额外出现对得分贡献的递减速度。

- **较低的 k（如 0.5）**：快速达到饱和 - 适合短文档
- **较高的 k（如 2.0）**：较慢饱和 - 适合长文档
- **典型值**：k = 1.2

**效果**：从 2 次出现增加到 4 次的影响，比从 200 次增加到 202 次的影响更大。

#### 2. 文档长度归一化（b 参数）

b 参数调整文档长度的惩罚程度。

- **b = 0**：无长度归一化（长度不重要）
- **b = 1**：完全长度归一化（严重惩罚长文档）
- **典型值**：b = 0.75

**直觉理解**：一个 10 词的文档中出现 1 次关键词，比一个 1000 词的文档中出现 10 次关键词更有说服力。

#### 3. IDF 组成部分

BM25 使用改进的 IDF 公式：

$$\text{IDF}(q_i) = \log\left(\frac{N - \text{DF}(q_i) + 0.5}{\text{DF}(q_i) + 0.5} + 1\right)$$

其中：

- $N$ = 语料库中的总文档数
- $\text{DF}(q_i)$ = 包含词项 $q_i$ 的文档数

加 1 是为了防止当词项出现在超过一半的语料库中时出现负值。

### 参数调优指南

**k 参数的调整：**

- 问问自己："我的文档平均有多长？"
- 长文档（论文、文章）：k = 1.5 - 2.0
- 中等文档（产品描述）：k = 1.2（默认值）
- 短文档（标题、推文）：k = 0.5 - 1.0

**b 参数的调整：**

- 问问自己："在我的领域中，文档长度是否表明相关性？"
- 科学/技术文档：b = 0.3 - 0.5（长度不太重要）
- 一般网页内容：b = 0.75（默认值）
- 社交媒体/评论：b = 0.8 - 0.9（更多地惩罚长度）

**注意**：这些是起点。最优值遵循"没有免费午餐"定理 - 需要在特定语料库上测试。

## BM25 vs TF-IDF 对比

| 方面           | TF-IDF                       | BM25                           |
| -------------- | ---------------------------- | ------------------------------ |
| **相关性评分** | 词频线性增长                 | 非线性饱和增长                 |
| **文档长度**   | 无归一化（偏向长文档）       | 使用 b 参数归一化              |
| **精确度**     | 适合均匀数据集               | 更适合变化大/噪声数据          |
| **长文档处理** | 偏向更长文档                 | 通过归一化公平处理             |
| **复杂度**     | 简单、快速                   | 稍微复杂一些                   |
| **最适合**     | 小数据集、精确匹配、可解释性 | 多样化数据集、长文档、生产搜索 |

## BM25 检索完整流程

### 从 Query 到 Chunk 的匹配过程

BM25 检索是一个多步骤的过程，涉及查询处理、文档分块、匹配和相关性评分。以下详细说明整个流程：

#### 步骤 1: 查询预处理

当用户输入查询时，首先需要对查询进行预处理：

```python
# 原始查询
query = "如何提高大语言模型在检索增强生成系统中的准确率？"

# 1.1 分词
tokens = tokenizer.tokenize(query)
# 结果: ["如何", "提高", "大", "语言", "模型", "在", "检索", "增强",
#        "生成", "系统", "中", "的", "准确率", "？"]

# 1.2 停用词过滤
stopwords = {"如何", "在", "中", "的", "？"}
filtered_tokens = [t for t in tokens if t not in stopwords]
# 结果: ["提高", "大", "语言", "模型", "检索", "增强", "生成", "系统", "准确率"]

# 1.3 词干提取/词形归一化（可选）
# 中文通常不需要，英文示例：
# "running" → "run", "better" → "good"

# 1.4 查询词项列表
query_terms = filtered_tokens
# 最终查询词项: ["提高", "大", "语言", "模型", "检索", "增强", "生成", "系统", "准确率"]
```

**关键点：**

- 分词器的选择很重要（jieba、HanLP 等）
- 停用词列表影响检索效果
- 保留的词项将用于后续匹配

#### 步骤 2: 文档分块（Chunking）

在 RAG 系统中，长文档需要切分成多个 Chunk 以提高检索精度：

```python
# 2.1 原始文档
document = """
大语言模型（LLM）在检索增强生成（RAG）系统中扮演着核心角色。
通过将外部知识库与生成能力结合，RAG 能够提供更准确、更时效的回答。

提高 RAG 系统准确率的关键方法包括：
1. 优化检索策略：使用混合检索结合稀疏和密集向量
2. 改进 Chunk 分割：根据语义边界切分文档
3. 引入重排序：使用 Cross-Encoder 对初步结果重排序
"""

# 2.2 分块策略
def chunk_document(text, chunk_size=200, overlap=50):
    """
    按字符数切分，保持重叠以避免信息丢失
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

chunks = chunk_document(document)

# 2.3 Chunk 示例
# Chunk 1: "大语言模型（LLM）在检索增强生成（RAG）系统中扮演着核心角色。
#          通过将外部知识库与生成能力结合，RAG 能够提供更准确、更时效的回答。"
# Chunk 2: "提高 RAG 系统准确率的关键方法包括：
#          1. 优化检索策略：使用混合检索结合稀疏和密集向量"
# Chunk 3: "2. 改进 Chunk 分割：根据语义边界切分文档
#          3. 引入重排序：使用 Cross-Encoder 对初步结果重排序"
```

**分块策略对比：**

| 策略              | 优点             | 缺点               | 适用场景               |
| :---------------- | :--------------- | :----------------- | :--------------------- |
| **固定长度**      | 实现简单、速度快 | 可能切断语义完整性 | 结构化文档             |
| **句子边界**      | 保持语义完整     | Chunk 长度不均     | 自然语言文本           |
| **段落边界**      | 保持逻辑完整     | 可能过长或过短     | 有明确段落的文档       |
| **语义切分**      | 最优语义保留     | 计算成本高         | 高质量要求场景         |
| **滑动窗口+重叠** | 避免边界信息丢失 | 存储空间增加       | 关键信息不能遗漏的场景 |

#### 步骤 3: Chunk 预处理和索引构建

对每个 Chunk 进行预处理并构建倒排索引：

```python
# 3.1 对每个 Chunk 进行预处理
chunk_1 = "大语言模型（LLM）在检索增强生成（RAG）系统中扮演着核心角色。"

# 分词
tokens_chunk1 = ["大", "语言", "模型", "LLM", "在", "检索", "增强",
                  "生成", "RAG", "系统", "中", "扮演", "核心", "角色"]

# 停用词过滤
filtered_chunk1 = ["大", "语言", "模型", "LLM", "检索", "增强",
                    "生成", "RAG", "系统", "扮演", "核心", "角色"]

# 3.2 计算词频 (TF)
from collections import Counter
tf_chunk1 = Counter(filtered_chunk1)
# 结果: {"检索": 1, "系统": 1, "大": 1, "语言": 1, "模型": 1, ...}

# 3.3 构建倒排索引
inverted_index = {
    "检索": [
        {"chunk_id": 1, "tf": 1, "positions": [5]},
        {"chunk_id": 2, "tf": 2, "positions": [3, 12]},
    ],
    "系统": [
        {"chunk_id": 1, "tf": 1, "positions": [9]},
        {"chunk_id": 2, "tf": 1, "positions": [5]},
    ],
    "准确率": [
        {"chunk_id": 2, "tf": 1, "positions": [8]},
        {"chunk_id": 5, "tf": 1, "positions": [2]},
    ],
    # ... 其他词项
}
```

**倒排索引说明：**

- 键：词项（term）
- 值：包含该词的 Chunk 列表及其元数据
- `positions`：词在 Chunk 中的位置（可用于短语查询）

#### 步骤 4: 计算 IDF（逆文档频率）

在所有 Chunks 处理完成后，计算每个词项的 IDF：

```python
import math

# 4.1 统计文档频率 (DF)
total_chunks = 100  # 假设总共有 100 个 chunks
df = {
    "检索": 45,      # 出现在 45 个 chunks 中
    "系统": 60,      # 出现在 60 个 chunks 中（更常见）
    "准确率": 15,    # 出现在 15 个 chunks 中（相对罕见）
    "大": 80,
    "语言": 75,
    "模型": 50,
}

# 4.2 计算 IDF
def calculate_idf(total_docs, df_term):
    """
    IDF = log((N - df + 0.5) / (df + 0.5) + 1)
    N: 总文档数
    df: 包含该词项的文档数
    """
    return math.log((total_docs - df_term + 0.5) / (df_term + 0.5) + 1)

idf = {term: calculate_idf(total_chunks, df[term]) for term in df}

# 结果示例:
# {
#     "检索": 0.81,      # 中等频率
#     "系统": 0.51,      # 高频，权重低
#     "准确率": 1.89,    # 低频，权重高（更有区分度）
#     "大": 0.22,
#     "语言": 0.29,
#     "模型": 0.69,
# }
```

**IDF 特性：**

- 常见词（如"系统"）IDF 值低
- 罕见词（如"准确率"）IDF 值高
- 反映词项的区分能力

#### 步骤 5: BM25 评分计算

对查询和每个 Chunk 计算 BM25 得分：

```python
def bm25_score(query_terms, chunk_tokens, chunk_length, avg_chunk_length,
               tf_dict, idf_dict, k1=1.5, b=0.75):
    """
    计算 BM25 得分

    参数:
    - query_terms: 查询词项列表
    - chunk_tokens: chunk 的词项列表
    - chunk_length: chunk 的长度（词数）
    - avg_chunk_length: 所有 chunks 的平均长度
    - tf_dict: chunk 中的词频字典
    - idf_dict: 全局 IDF 字典
    - k1: 调节词频饱和度 (通常 1.2-2.0)
    - b: 调节长度归一化 (通常 0.75)
    """
    score = 0.0

    # 对查询中的每个词项计算得分
    for term in query_terms:
        if term not in tf_dict:
            continue  # 该词不在 chunk 中，跳过

        # 5.1 获取词频和 IDF
        tf = tf_dict[term]
        idf = idf_dict.get(term, 0)

        # 5.2 长度归一化
        norm_length = 1 - b + b * (chunk_length / avg_chunk_length)

        # 5.3 计算该词项的得分
        term_score = idf * (tf * (k1 + 1)) / (tf + k1 * norm_length)

        score += term_score

    return score

# 示例计算
query_terms = ["提高", "检索", "准确率"]

# Chunk 1 数据
chunk1_tokens = ["大", "语言", "模型", "检索", "系统", "核心"]
chunk1_tf = {"检索": 1, "大": 1, "语言": 1, "模型": 1, "系统": 1, "核心": 1}
chunk1_length = len(chunk1_tokens)

# Chunk 2 数据
chunk2_tokens = ["提高", "准确率", "检索", "策略", "检索", "向量"]
chunk2_tf = {"提高": 1, "准确率": 1, "检索": 2, "策略": 1, "向量": 1}
chunk2_length = len(chunk2_tokens)

avg_length = 15  # 假设平均 chunk 长度

# 计算得分
score_chunk1 = bm25_score(query_terms, chunk1_tokens, chunk1_length,
                          avg_length, chunk1_tf, idf)
score_chunk2 = bm25_score(query_terms, chunk2_tokens, chunk2_length,
                          avg_length, chunk2_tf, idf)

print(f"Chunk 1 得分: {score_chunk1:.4f}")  # 输出: 0.81 (只匹配"检索")
print(f"Chunk 2 得分: {score_chunk2:.4f}")  # 输出: 4.32 (匹配所有三个词)
```

**评分详解：**

以 Chunk 2 为例，计算过程：

```
查询词: ["提高", "检索", "准确率"]

词项 "提高":
  - tf = 1, idf = 1.2
  - norm = 1 - 0.75 + 0.75 * (6/15) = 0.55
  - score = 1.2 * (1 * 2.5) / (1 + 1.5 * 0.55) = 1.63

词项 "检索":
  - tf = 2, idf = 0.81
  - norm = 0.55
  - score = 0.81 * (2 * 2.5) / (2 + 1.5 * 0.55) = 1.46

词项 "准确率":
  - tf = 1, idf = 1.89
  - norm = 0.55
  - score = 1.89 * (1 * 2.5) / (1 + 1.5 * 0.55) = 2.57

总分 = 1.63 + 1.46 + 2.57 = 5.66
```

#### 步骤 6: 排序和返回 Top-K

根据 BM25 得分对所有 Chunks 排序并返回最相关的结果：

```python
# 6.1 收集所有 Chunk 的得分
chunk_scores = []
for chunk_id, chunk_data in enumerate(all_chunks):
    score = bm25_score(
        query_terms,
        chunk_data['tokens'],
        chunk_data['length'],
        avg_length,
        chunk_data['tf'],
        idf
    )
    chunk_scores.append({
        'chunk_id': chunk_id,
        'score': score,
        'text': chunk_data['text'],
        'metadata': chunk_data['metadata']
    })

# 6.2 按得分降序排序
chunk_scores.sort(key=lambda x: x['score'], reverse=True)

# 6.3 返回 Top-K
top_k = 5
results = chunk_scores[:top_k]

# 6.4 输出结果
for rank, result in enumerate(results, 1):
    print(f"排名 {rank}: (得分: {result['score']:.4f})")
    print(f"  Chunk ID: {result['chunk_id']}")
    print(f"  内容: {result['text'][:100]}...")
    print(f"  元数据: {result['metadata']}")
    print()
```

**输出示例：**

```
排名 1: (得分: 5.66)
  Chunk ID: 2
  内容: 提高 RAG 系统准确率的关键方法包括：1. 优化检索策略：使用混合检索结合稀疏和密集向量...
  元数据: {'source': 'rag_guide.pdf', 'page': 3, 'section': '优化方法'}

排名 2: (得分: 3.21)
  Chunk ID: 7
  内容: 检索准确率的评估指标包括 Precision@K、Recall@K 和 MRR。通过这些指标可以量化...
  元数据: {'source': 'rag_guide.pdf', 'page': 8, 'section': '评估指标'}

排名 3: (得分: 2.15)
  Chunk ID: 12
  内容: 在实际应用中，检索系统的性能优化需要平衡准确率和速度。索引结构的选择...
  元数据: {'source': 'rag_guide.pdf', 'page': 15, 'section': '性能优化'}
```

### 完整流程总结

```
用户查询
    ↓
[步骤 1] 查询预处理
    - 分词: "提高检索准确率" → ["提高", "检索", "准确率"]
    - 停用词过滤
    - 词形归一化
    ↓
[步骤 2] 文档分块
    - 长文档 → 多个 Chunks
    - 保持语义完整性
    - 可选：添加重叠区域
    ↓
[步骤 3] Chunk 预处理
    - 对每个 Chunk 分词
    - 计算词频 (TF)
    - 构建倒排索引
    ↓
[步骤 4] 计算 IDF
    - 统计词项在多少 Chunks 中出现
    - 计算逆文档频率
    - 罕见词获得更高权重
    ↓
[步骤 5] BM25 评分
    - 对每个 Chunk 计算相关性得分
    - 考虑：词频、IDF、文档长度
    - 公式：BM25(q,d) = Σ IDF(qi) × f(qi,d)×(k1+1) / [f(qi,d) + k1×(1-b+b×|d|/avgdl)]
    ↓
[步骤 6] 排序返回
    - 按得分降序排列所有 Chunks
    - 返回 Top-K 最相关的 Chunks
    - 附带元数据（来源、页码等）
    ↓
检索结果
```

### 影响检索质量的关键因素

1. **分词质量**
   - 中文：jieba、HanLP、LTP
   - 英文：NLTK、spaCy
   - 影响：分词错误会导致匹配失败

2. **Chunk 大小**
   - 太小：上下文不足，语义不完整
   - 太大：噪音多，相关性降低
   - 推荐：200-500 tokens，根据场景调整

3. **停用词列表**
   - 通用停用词：的、了、在、是
   - 领域停用词：根据具体领域定制
   - 影响：过度过滤会丢失关键信息

4. **参数调优**
   - `k1` (1.2-2.0)：控制词频饱和度
   - `b` (0.75)：控制长度归一化
   - 根据数据集特点调整

5. **索引优化**
   - 倒排索引压缩
   - 缓存热点查询
   - 增量更新策略

## 在 Milvus 中实现 BM25

### 环境设置

```python
from pymilvus import MilvusClient, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction

# 初始化分析器（处理停用词、分词、统计）
analyzer = build_default_analyzer(language="en")

# 创建 BM25 嵌入函数
bm25_ef = BM25EmbeddingFunction(analyzer)

# 在语料库上拟合以收集词项统计信息
corpus = ["文档 1 文本", "文档 2 文本", ...]
bm25_ef.fit(corpus)
```

### 创建稀疏向量

```python
# 将文档编码为稀疏向量
documents = ["产品标题 1", "产品标题 2", ...]
document_embeddings = bm25_ef.encode_documents(documents)

# 检查维度
print(f"稀疏维度: {bm25_ef.dim}")  # = 语料库中唯一词项的总数
print(f"向量形状: {document_embeddings[0].shape}")
```

### 设置 Milvus 集合

```python
# 定义 schema
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True,
                auto_id=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
]

schema = CollectionSchema(fields, "BM25 稀疏检索")
collection = Collection("bm25_collection", schema)

# 为稀疏向量创建索引
sparse_index = {
    "index_type": "SPARSE_INVERTED_INDEX",  # 对高维稀疏向量高效
    "metric_type": "IP"  # 内积（目前唯一支持的度量）
}
collection.create_index("sparse_vector", sparse_index)

# 插入数据
entities = [documents, document_embeddings]
collection.insert(entities)
collection.flush()
```

### 执行搜索

```python
# 编码查询
queries = ["我应该买什么产品用于旅行"]
query_embeddings = bm25_ef.encode_queries(queries)

# 搜索
collection.load()
results = client.search(
    collection_name="bm25_collection",
    data=query_embeddings[0],
    anns_field="sparse_vector",
    limit=5,
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["text"]
)

# 结果包含按 BM25 得分排序的最相关文档
```

## 稀疏向量索引方法

### 倒排索引（Inverted Index）

**最适合**：高维稀疏向量（如 BM25 输出）

**工作原理：**

- 将每个维度映射到具有非零值的文档
- 在搜索期间提供对相关数据的直接访问
- 类似于搜索引擎索引网页的方式

**优势：**

- 检索速度非常快
- 对稀疏数据的内存效率高
- 随语料库大小良好扩展

### WAND（Weak-AND）算法

稀疏向量搜索的**替代方法**，可以提前跳过非有希望的候选项。

## 最佳实践

### 何时使用 BM25

✅ **适合的使用场景：**

- 基于关键词的搜索系统
- 电商产品搜索
- 具有特定术语的文档检索
- 学术论文搜索
- 法律/医疗文档检索
- 需要可解释结果时

❌ **不理想的场景：**

- 语义相似性搜索（使用密集向量）
- 查询和文档使用不同词汇时
- 跨语言搜索
- 仅靠精确关键词匹配不够时

### 优化技巧

1. **预处理至关重要：**
   - 根据领域适当删除停用词
   - 考虑词干提取/词形还原
   - 处理特殊字符和标点符号
2. **语料库统计很重要：**
   - 当语料库发生重大变化时重新拟合 BM25
   - 监控重要词项的 IDF 分数
3. **与其他方法结合：**
   - 使用混合搜索（BM25 + 密集向量）以获得最佳结果
   - 添加元数据过滤
   - 实现重排序阶段

4. **性能考虑：**
   - BM25 速度快，但稀疏向量可能很大
   - 使用倒排索引提高效率
   - 对于超大规模考虑近似方法

## 混合搜索策略

BM25 稀疏向量与神经模型的密集向量结合使用效果极佳：

```python
# 结合 BM25（稀疏）和神经嵌入（密集）
# - BM25：擅长精确关键词匹配
# - 密集向量：擅长语义相似性

# 典型方法：
# 1. 使用 BM25 检索候选文档（快速、基于关键词）
# 2. 使用密集向量重排序（语义理解）
# 3. 或使用 RRF（倒数排名融合）组合分数
```

**混合方法的好处：**

- 结合词法和语义匹配
- 更好的召回率和精确度
- 对词汇不匹配更稳健
- 两全其美

## 进阶考虑

### 稀疏向量属性

- **维度**：等于词汇表中唯一词项的总数（可能超过 10 万）
- **稀疏性**：通常 >99% 为零（只有查询词项具有非零值）
- **可解释性**：每个维度对应一个特定的词项
- **存储**：以稀疏格式存储时高效

### 与 SPLADE 的比较

**BM25：**

- 纯统计，无神经网络
- 快速，无需 GPU
- 仅原始查询词项获得分数
- 非常适合精确匹配

**SPLADE：**

- 基于神经网络（BERT 基础）
- 可以包含语义相似的词项
- 更好的词汇扩展
- 计算成本更高

## 资源与参考

- 原始博客文章：[Mastering BM25 on Zilliz](https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus)
- Milvus 稀疏向量文档
- 概率检索模型学术论文

## 总结

BM25 是一种强大、可解释的稀疏检索算法，具有以下特点：

- 通过饱和度和长度归一化改进了 TF-IDF
- 使用可调参数（k, b）适应不同使用场景
- 与 Milvus 等向量数据库无缝集成
- 为检索系统提供强大的基线
- 在混合方法中与密集向量很好地结合

对于生产级 RAG 系统，建议将 BM25 作为词法检索组件，与语义（密集向量）检索一起使用，以获得最佳结果。
