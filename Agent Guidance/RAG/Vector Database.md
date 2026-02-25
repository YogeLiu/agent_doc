# Vector Database：从数学原理到工程落地（后端转型 Agent / RAG 视角）

这篇文档把「向量数据库/向量检索」从技术原理讲到工程落地，目标是让你能：

- 解释清楚 embedding、相似度度量、ANN（Approximate Nearest Neighbor）的数学关系
- 在不同数据规模、延迟预算、内存预算下做出合理选型（Flat / HNSW / IVF / PQ / IMI）
- 理解向量库在生产系统里的真实形态：索引、存储、过滤、更新、分片与评估

---

## 1. Embedding（向量表示）

### 1.1 Embedding 的本质：把语义映射到向量空间

给定一个输入 $x$（文本/图片/代码/用户画像等），embedding 模型 $f(\cdot)$ 输出一个向量：

$$
\mathbf{v} = f(x) \in \mathbb{R}^{d}
$$

你希望“语义相近”的样本在向量空间里“距离更近/相似度更高”。向量数据库做的事情本质上是：

 - 存：存大量 $\mathbf{v}_i$（以及元数据）
 - 找：给定 query 向量 $\mathbf{q}$，在 $N$ 个向量里找 Top‑k 最近邻

### 1.2 Embedding 维度、范数与归一化

- 维度 $d$：越大表达能力通常越强，但存储/计算更贵（计算通常是 $O(d)$）。
- 范数 $\|\mathbf{v}\|$：不同模型输出的范数分布可能差异很大；范数会影响 dot product 与 L2 的意义。
- 归一化（normalize）：

$$
\hat{\mathbf{v}}=\frac{\mathbf{v}}{\|\mathbf{v}\|_2}
$$

把所有向量都归一化到单位球面后，“角度/方向”成为主要信息，cosine 与 L2 的关系变得非常干净（见下文 2.3）。

工程经验（尤其在 RAG）：
- 如果用 cosine similarity，通常会统一做 normalize，避免范数差异造成偏置。
- 如果用 dot product（MIPS），是否 normalize 取决于你的语义定义：你到底希望“方向相近”还是“方向相近且范数大更重要”。

---

## 2. Similarity（相似度/距离度量）

向量检索中，常用的度量包括：

- L2 距离（Euclidean distance）
- 内积（Dot product / Inner product）
- 余弦相似度（Cosine similarity）

很多“看起来不同”的度量，在某些条件下可以互相转化，这是理解“为什么某些库默认用 L2 也能做 cosine”的关键。

### 2.1 L2 Distance（欧氏距离）

定义：

$$
d_2(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_2
$$

平方形式更常用于推导：

$$
\|\mathbf{x}-\mathbf{y}\|_2^2
=\|\mathbf{x}\|_2^2+\|\mathbf{y}\|_2^2-2\mathbf{x}\cdot\mathbf{y}
$$

重要结论：若所有向量范数相同（尤其是都被 normalize 到 1），那么最小 L2 等价于最大 dot / cosine（见 2.3）。

### 2.2 Dot Product（内积 / MIPS）

定义：

$$
s(\mathbf{x},\mathbf{y})=\mathbf{x}\cdot\mathbf{y}=\sum_{i=1}^{d}x_i y_i
$$

MIPS 的目标是：

$$
\arg\max_{\mathbf{x}\in\mathcal{D}} \mathbf{q}\cdot\mathbf{x}
$$

直觉：dot 同时考虑“方向”和“长度”（范数）。如果不 normalize，范数大的向量天然更容易得到更大的内积。

### 2.3 Cosine Similarity / Cosine Distance（余弦）

余弦相似度定义：

$$
\cos(\theta)=\frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|_2\|\mathbf{y}\|_2}
$$

当向量都归一化（$\|\mathbf{x}\|=\|\mathbf{y}\|=1$）：

$$
\|\mathbf{x}-\mathbf{y}\|_2^2 = 2 - 2\cos(\theta)
$$

因此，最大化 cosine 等价于最小化 L2 距离（平方）。这解释了很多系统里“用 L2 索引也能做 cosine”的原因。

### 2.4 选择度量的工程建议

- 文本语义检索/RAG：更常见的是 cosine（配合 normalize），稳定且直觉清晰。
- 推荐/广告向量：dot product 也很常见（范数承载“强度/置信度/热度”等信息）。
- 落地时的关键：在 embedding 生成阶段和向量库索引度量保持一致，否则会出现“理论相似但检索不相似”的错觉。

### 2.5 MIPS 如何转为 L2（为什么很多索引更爱 L2）

很多 ANN 索引的理论/实现更适配 L2；但实际可能需要做 MIPS（dot product）。常见做法是把 MIPS 转成 L2 NN。

一种经典的向量增广（假设数据向量范数有上界 \( \|\mathbf{x}\|\le R\)）：

$$
\mathbf{x'} = [\mathbf{x}; \sqrt{R^2 - \|\mathbf{x}\|_2^2}],\quad
\mathbf{q'} = [\mathbf{q}; 0]
$$

可证明：

$$
\|\mathbf{q'}-\mathbf{x'}\|_2^2
= \|\mathbf{q}\|_2^2 + R^2 - 2\mathbf{q}\cdot\mathbf{x}
$$

在固定 $\mathbf{q}$ 与 $R$ 的情况下，最小化 L2 距离等价于最大化 $\mathbf{q}\cdot\mathbf{x}$。

---

## 3. Similarity Search（相似度检索）

给定向量集合 $\{\mathbf{x}_i\}_{i=1}^N$ 和 query 向量 $\mathbf{q}$，检索 Top‑k 的数学形式是：
$$
\text{TopK}(\mathbf{q})=\operatorname{arg\,topk}_{i\in[1..N]}\ \text{sim}(\mathbf{q},\mathbf{x}_i)
$$

其中 $\text{sim}$ 可以是 cosine/dot，或者用距离 $d$ 的负数。

### 3.1 精确搜索（Exact NN）

#### 3.1.1 Brute Force（Flat 扫描）

做法：对每个 $\mathbf{x}_i$ 计算一次相似度，取 Top‑k。

- 时间复杂度：$O(Nd)$
- 优点：精确；实现最简单；没有建索引时间
- 缺点：数据大时延迟不可控

工程上 Flat 常被低估：
- 小规模（<100k）非常好用
- 即使规模更大，配合 SIMD/GPU 与批量查询也能很强

#### 3.1.2 KNN（k‑Nearest Neighbors）

KNN 是“结果定义”（找最近的 k 个），不等于某个具体算法。Brute force 是实现 KNN 的一种方式；ANN 则是常用的近似实现方式。

### 3.2 近似搜索（ANN：Approximate NN）

ANN 的目标是在可接受的 recall 损失下，显著降低查询时间/内存成本。

你需要掌握的核心 trade‑off：
- Recall vs Latency：更高的召回通常意味着更慢
- Memory vs Latency：图索引/高参数通常更快但更吃内存
- Build Cost vs Query Cost：某些索引训练/构建昂贵但查询便宜（如 IVF/PQ）

---

## 4. ANN 的主流算法族（按索引思想分类）

业界常见 ANN 可以按索引结构分为四类：

1) 空间划分（KD‑Tree / Ball‑Tree 等）  
2) 聚类 + 倒排（IVF / IMI）  
3) 图结构（NSW / HNSW）  
4) 压缩与量化（PQ / OPQ / SQ / int8 等，常与 IVF 组合）

### 4.1 基于空间划分

#### 4.1.1 KD‑Tree

思想：递归按某一维把空间二分，形成树；查询时用回溯剪枝减少访问节点。

- 适用维度：低维效果好；高维退化严重（维度灾难）
- 工程结论：现代 embedding 常见 384/768/1024 维，KD‑Tree 通常不作为主力 ANN（除非做特征降维或任务是低维）。

#### 4.1.2 Ball Tree

思想：用球（中心 + 半径）包住一组点，查询时用三角不等式剪枝。

相较 KD‑Tree，对某些数据分布更友好，但高维同样会退化。

### 4.2 基于聚类划分：IVF / IMI

这条路线的关键词是：先粗召回，再细计算。

#### 4.2.1 IVF（Inverted File Index）

核心思想：
 - 先训练一个粗量化器（coarse quantizer），把空间划分成 $n_{\text{list}}$ 个簇（centroids）
 - 每个向量 $\mathbf{x}$ 被分配到最近的 centroid 对应的倒排表（list）中
 - 查询时只探测 $n_{\text{probe}}$ 个最相关的 lists

关键参数：
- nlist：簇的数量（倒排桶数量）
  - nlist 大：每个桶更小，查询更快，但训练/构建成本更高，且需要更精细的探测策略
- nprobe：查询时探测多少个桶
  - nprobe 大：recall 高但更慢

复杂度直觉：
 - 把全库从 $N$ 缩小到候选集大小 $C\approx N\cdot \frac{n_{\text{probe}}}{n_{\text{list}}}$，然后在候选集里再做更精确的距离计算。

工程常识：
- IVF 非常适合 1M+ 规模，并且能很好地和 PQ 组合（IVF‑PQ）。
- IVF 对训练数据分布较敏感：训练向量与线上向量分布偏移会导致 recall 降低。

#### 4.2.2 IMI（Inverted Multi‑Index）

IMI 可以理解为把粗量化器做成乘积结构，用多个子空间的笛卡尔积构造更细粒度的桶，从而在超大规模下获得更高的划分分辨率。

工程上：
- IMI 更偏“极大规模、强工程团队”的路线（训练/实现更复杂）
- 许多现代系统在可接受的硬件下，优先用「IVF‑PQ + 分片」而不是 IMI

### 4.3 基于图结构：NSW / HNSW

这条路线的关键词是：用图的“近邻导航”快速逼近目标。

#### 4.3.1 NSW（Navigable Small World）

将向量作为节点，边连接若干近邻；查询时从入口点开始做贪心搜索（不断走向更相近的节点），利用小世界网络的短路径性质快速到达近邻区域。

#### 4.3.2 HNSW（Hierarchical NSW）

HNSW 在 NSW 上增加多层结构：
- 上层更稀疏、跨度更大，用于快速跳转到大概区域
- 越往下层越稠密，用于精细搜索

关键参数（概念级）：
- M：每个节点的最大连接数（出度）
  - M 大：图更密、recall 更高/更快，但内存更大、构建更慢
- efConstruction：构建时的候选队列大小（越大越“精”但构建更慢）
- efSearch：查询时的候选队列大小（越大 recall 越高但更慢）

工程结论：
- HNSW 往往是中等到大规模（100k～千万级）+ 低延迟 + 内存充足场景的首选
- HNSW 本质是内存换时间，并且删除/更新在很多实现里需要额外策略（tombstone + 重建）

### 4.4 压缩与量化（Quantization）

目标：减少存储与带宽，同时尽量保留相似度排序的正确性。

常见手段：
- 标量量化（Scalar Quantization / int8）：把 float 压成 int8/int16，简单直接
- PQ（Product Quantization）：把向量切成多个子空间，每个子空间用码本索引表示
- OPQ（Optimized PQ）：在 PQ 前学习一个旋转，使得子空间更可量化

#### 4.4.1 PQ（Product Quantization）原理

将 $d$ 维向量切成 $m$ 段（每段 $d/m$ 维）：

$$
\mathbf{x}=[\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}]
$$

每段训练一个大小为 $k=2^b$ 的码本（centroids）。编码后每段只存一个索引（$b$ bits），整体 code 长度为 $m\cdot b$ bits。

压缩率直觉：
- 原始 float32：$d \cdot 4$ bytes
- PQ code：$\frac{m\cdot b}{8}$ bytes

例如 $d=768,m=96,b=8$：
- 原始：$768\cdot4=3072$ bytes
- PQ：$96\cdot8/8=96$ bytes（约 32 倍压缩）

PQ 的关键优势是可以用 ADC（Asymmetric Distance Computation）：查询向量不量化，库向量量化，用预计算表快速近似距离。

#### 4.4.2 IVF + PQ（IVF‑PQ）：最常见的“大规模标配”

IVF 解决候选集缩小，PQ 解决存储与距离计算加速。

典型流程：
- 训练 coarse centroids（IVF）
- 训练 PQ codebooks（通常对 residual 做 PQ：$\mathbf{x}-\mathbf{c}$）
- 查询：nprobe 个桶 -> PQ 快速算距离 -> Top‑k

---

## 5. 向量数据库在生产里长什么样（系统视角）

一个向量数据库通常包含以下子系统：

- Index（索引）：HNSW/IVF‑PQ/Flat 等
- Vector Store（向量存储）：原始向量或压缩向量的持久化（内存 + 磁盘）
- Metadata Store（元数据）：doc_id、chunk_id、时间、权限、标签、tenant 等
- Filtering（过滤）：基于元数据的查询约束（where / filter）
- Compute（计算）：SIMD/GPU、多线程、批处理
- Replication/Sharding（副本/分片）：扩展容量与 QPS
- Ingestion（写入流水线）：embedding 生成、批量写入、增量更新

### 5.1 过滤（Metadata Filter）是工程地狱的来源之一

向量检索最怕这句话：

> “我要在向量检索的同时按 tenant_id / 权限 / 时间范围过滤。”

原因：ANN 索引是按几何邻近组织的，过滤条件是按业务维度组织的，两者天然不一致。

常见策略：
- Post‑filter：先向量 Top‑k，再按元数据过滤，不够再扩大 Top‑k
  - 简单，但可能导致延迟不稳定、recall 降低
- Pre‑filter（bitset / posting list）：查询时只在满足条件的集合里做 ANN
  - 需要索引实现支持（例如对 HNSW 做 bitset 过滤/多图/分区）
- 按 tenant 分库/分 collection：最干净但资源碎片化

### 5.2 更新与删除：别被“支持 upsert”四个字骗了

不同索引对动态更新的友好度差异很大：
- Flat：最好（追加 + 删除标记都容易）
- HNSW：通常可插入；删除常用 tombstone，长期需要重建/重连边
- IVF‑PQ：追加可以做；但训练（nlist、PQ codebook）通常离线完成，分布漂移会影响效果

生产常见做法：
- 写入走增量 + 定期重建（nightly rebuild / rolling rebuild）
- 保留 WAL + snapshot，保证可恢复与一致性

---

## 6. 选型与规模经验（含推荐表）

下面是经验推荐，前提是 embedding 维度在数百量级（384/768），延迟目标在几十到几百毫秒。

|数据规模（向量数 N）|延迟预算|内存预算|推荐索引|为什么|
|---|---:|---:|---|---|
|<100k|宽松/中等|低/中|Flat（暴力）|实现简单、结果精确，SIMD 足够快|
|100k–1M|低|中/高|HNSW|低延迟强、参数可调、工程成熟|
|1M–10M|中/低|中|IVF‑PQ 或 HNSW（内存足够时）|IVF‑PQ 省内存；HNSW 更吃内存但更快|
|10M+|中/低|中/高|IVF‑PQ + 分片（必要时 GPU）|容量与吞吐更可控，便于水平扩展|
|超大规模（十亿级）|严苛|高|IMI + PQ / 多级索引 / 分层存储|训练与系统复杂度上升，需要更强工程化|

在做系统设计时，最好先明确四个数字：
- \(N\)：向量数
- \(d\)：向量维度
- SLA：P95/P99 延迟
- QPS：峰值并发

这四个数字会决定你是在“算法优化”还是“系统扩展”（分片/副本/GPU/缓存）上花主要精力。

---

## 7. 参数调优的抓手（面试能讲、生产能用）

### 7.1 HNSW：三个旋钮

- M：更大通常 recall 更高、查询更快，但内存更大
- efConstruction：构建质量（离线成本）
- efSearch：在线 recall/latency 的主旋钮

调优套路：
- 固定 M 与 efConstruction（离线建好）
- 线上只调 efSearch 以满足 SLA 与 recall 指标

### 7.2 IVF：nlist / nprobe 的直觉

- nlist：把空间切多细
- nprobe：查询时探测多少个桶

调优套路：
- nlist 先按规模选一个数量级（让平均桶大小在可控范围）
- 用 nprobe 在“延迟‑召回”曲线上选点

### 7.3 PQ：m 与 b 的直觉

- m（子空间数）：越大越精细（但计算/码本更大）
- b（每段 bits）：越大误差越小但 code 更长

PQ 调优本质是：在“内存/带宽/缓存命中”与“距离近似误差”之间找平衡。

---

## 8. RAG/Agent 场景下，向量库不是终点

向量库通常只是 RAG 的召回层。完整链路更像：

```text
Query
  ↓
（可选）Query Rewrite / 多查询扩展
  ↓
Hybrid Retrieval（BM25 + Vector）
  ↓
（可选）Rerank（Cross-Encoder / LLM）
  ↓
Context Assemble（去重、聚合、窗口控制）
  ↓
LLM Generate
  ↓
评估与反馈（线上指标/人工/LLM-as-judge）
```

实践经验：
- 向量召回要做“宽召回”（高 recall），把精度交给 rerank 与生成阶段
- Hybrid（关键词 + 向量）解决专有名词/数字/代码符号这类 dense 模型天然不擅长的问题
- 评估体系要同时有：
  - 检索指标：Recall@k、MRR、NDCG、延迟、吞吐
  - 生成指标：Faithfulness（忠实性）、Answer relevance、引用覆盖率

---

## 9. 你应该记住的三句话（背后的原理是什么）

1) “normalize 后，cosine≈L2” — 数学等价来自 \(\|\mathbf{x}-\mathbf{y}\|^2 = 2-2\cos\theta\)（单位向量）

2) “HNSW 是内存换时间” — 图越密、候选队列越大，越容易绕开局部最优，但存储边与访问候选都要付出成本

3) “IVF‑PQ 是大规模的性价比” — IVF 缩小候选集，PQ 压缩存储并加速距离估计，把“全量 O(Nd)”变成“少量候选 + 近似距离”
