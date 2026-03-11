# 重排序：让对的文档排在前面

向量检索召回 Top-K，得到的往往是一个"相关性还行但不够准"的列表。可能排第一的是相关文档，但也可能排第三、第五。后面的生成环节能不能用好这些召回结果，很大程度上取决于"谁排第一"。

重排序（rerank）要解决的就是这个问题：已经找到了候选文档，怎么排得更准？

---

## 为什么需要 rerank

向量检索用的是 Bi-encoder 架构——query 和 document 分别独立编码成向量，然后在向量空间里算相似度。

这个架构有个天然的缺陷：query 和 document 在编码时"没有真正见过对方"。

举例来说：

- Query："怎么提高 RAG 系统的准确率"
- Document A："本文介绍 RAG 系统的评估方法，包括 Recall、MRR 等指标"
- Document B："提高 RAG 准确率的核心是优化检索策略，包括混合检索、query 改写等技术"

用 Bi-encoder 分别编码，A 和 B 的向量可能差不多"准确"——都和 query 有点相关。但 B 明显更直接回答了"怎么提高"这个问题。

问题在于：Bi-encoder 编码时不知道 query 具体在问什么，只能靠"语义方向大致对"来匹配。

这就是 rerank 存在的意义。rerank 用 Cross-encoder 架构——把 query 和 document 拼在一起输入模型，让模型同时"看到"两边，真正做一次"query-document 交互式"的打分。

---

## Bi-encoder vs Cross-encoder

这是两种本质不同的编码架构，理解它们的区别是选型的基础。

### Bi-encoder（双编码器）

```
Query → Encoder_Q → 向量 v_q
Doc → Encoder_D → 向量 v_d

相似度 = sim(v_q, v_d)
```

特点：

- Query 和 Document 分别独立编码
- 可以预先算好所有文档的向量，存库里
- 查询时只编码 query，然后做向量检索——快

缺点：

- 编码时 query 和 doc 没有"见过面"
- 只能捕获粗粒度的语义相似，无法捕捉细粒度的相关性
- 对于"同一个意思不同说法"的匹配能力弱

### Cross-encoder（交叉编码器）

```
[CLS] Query [SEP] Document [SEP] → Encoder → 打分 s
```

特点：

- Query 和 Document 拼接后一起输入模型
- 模型在每一层都能做 query 和 doc 的注意力交互
- 打分更精准，能捕捉细粒度的相关性

缺点：

- 每个 (query, doc) pair 都要单独过一次模型——慢
- 无法预先存文档向量（因为向量依赖于具体 query）
- 计算成本是 O(N)，N 是候选文档数

### 对比

| 维度     | Bi-encoder      | Cross-encoder   |
| -------- | --------------- | --------------- |
| 速度     | 快（O(1) 查表） | 慢（O(N) 推理） |
| 精度     | 粗              | 细              |
| 预计算   | 可以            | 不行            |
| 适用规模 | 大规模召回      | 小规模精排      |

---

## 两阶段策略：召回 + 精排

既然 Bi-encoder 快但不准，Cross-encoder 准但不快，最好的方案是结合起来用：

```
Query → Bi-encoder 召回 Top-100 → Cross-encoder 重排 Top-10 → LLM 生成
```

这就是工业界最常用的两阶段策略：

1. **第一阶段（召回）**：用 Bi-encoder 快速召回候选集合。目标是"相关文档别漏掉"，可以召回得多一些（比如 Top 50-100）。

2. **第二阶段（精排）**：用 Cross-encoder 对候选集合重排序。目标是"最相关的排到前面"，取 Top 5-10 送给 LLM。

这个策略在效果和成本之间取得了很好的平衡。真正消耗计算资源的是第二阶段（Cross-encoder），但候选数已经从 100 万缩小到 100，成本可控。

---

## 主流重排序模型

### 开源模型

**BGE-Reranker 系列**

BAAI 出品的 reranker 模型系列：

- bge-reranker-base：轻量版，效果够用
- bge-reranker-large：效果更好，但更慢、更占显存
- bge-reranker-v2：最新版本，支持更多语言和更长文本

实测结论：bge-reranker-large 在中文场景效果很好，是开源 reranker 的首选。

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

# 打分
score = reranker.compute_score(["query", "document"])
# 或者批量
scores = reranker.compute_score_batch([["query", "doc1"], ["query", "doc2"]])
```

**Cohere Rerank**

Cohere 提供的 rerank API。支持 100+ 语言，效果公认不错。

```python
from cohere import Client

co = Client(api_key="your-key")
response = co.rerank(
    query="your query",
    documents=["doc1", "doc2", "doc3"],
    top_n=3,
    model="rerank-multilingual-v3.0"
)
```

优点：不需要自己托管模型，效果稳定。缺点：数据要出网，有延迟和成本。

**Zotero/Flashrank**

轻量级的 reranker 模型，可以在 CPU 上跑，适合延迟敏感场景。

### API 服务

**Cohere Rerank API**

上面提到了，按调用次数收费。适合不想运维模型的团队。

**bge-reranker API**

很多云服务商提供基于 bge-reranker 的托管 API，比如 OpenAI 旗下的某些服务。

### 模型选择建议

- 效果优先：bge-reranker-large 或 Cohere Rerank API
- 成本优先：bge-reranker-base + int8 量化
- 延迟优先：Flashrank 或直接用 API

---

## 工程实践中的几个问题

### 1. 重排多少候选？

这是第一个要决策的问题。重排 Top-K，K 设多大？

设小了，可能把相关文档漏掉；设大了，延迟和成本上升。

经验值：

- 召回阶段返回 50-100 个候选
- 重排阶段取 Top 10-20 送给 LLM

具体数值取决于：

- 召回质量。如果召回本身就好（Recall@100 已经很高），K 可以设小一点。
- 延迟预算。如果对延迟敏感，K 设小。
- LLM 上下文窗口。如果 LLM 支持的上下文很大，K 可以设大。

### 2. 延迟怎么控制

重排序的延迟主要是 Cross-encoder 推理的延迟。

优化方向：

- **批量推理**：一次处理多个 (query, doc) pair，比逐个处理快很多
- **模型量化**：int8 量化能减少 30-40% 延迟，效果损失很小
- **GPU 加速**：如果用开源模型，上 GPU 是必须的
- **缓存**：对于高频 query，可以缓存 rerank 结果

典型延迟参考（bge-reranker-large + A10G GPU）：

- 重排 50 个 doc：~100ms
- 重排 100 个 doc：~180ms

### 3. 需要训练数据吗

开源的 reranker 模型通常不需要自己训练——直接用预训练好的就能有效果。

但如果有条件，用自己的业务数据做进一步微调，效果通常能提升明显。

微调数据格式：

```json
[
  {
    "query": "如何提高RAG系统的准确率",
    "positive": ["提高RAG准确率的核心是优化检索策略..."],
    "negative": ["RAG系统的基本原理是..."]
  }
]
```

positive 是正样本（相关文档），negative 是负样本（不相关文档）。用 1000-2000 条这种数据就能做一次有效微调。

### 4. 重排结果怎么和召回结果融合

最简单的做法：直接用重排结果覆盖召回结果。

复杂一点的玩法：把重排分数和召回分数做加权融合。

```python
# 融合公式
final_score = α * rerank_score + (1 - α) * recall_score

# α 通常在 0.5-0.8 之间，重排分数权重更高
```

什么场景需要融合：当召回本身也比较准，只是排序不够精细时。融合能保留两边的优势。

---

## 何时需要 rerank，何时不需要

### 需要 rerank 的场景

- **召回质量不够好**：Top-10 里经常混进不相关或弱相关的文档
- **对排序精度要求高**：比如搜索结果的第一个结果直接影响用户满意度
- **有足够的延迟预算**：能接受增加 100-200ms 的延迟

### 不需要 rerank 的场景

- **召回已经足够好**：Recall@10 > 90%，Top-3 几乎都是对的
- **延迟极其敏感**：比如 < 100ms 的查询响应要求
- **成本极其敏感**：不想额外花 reranker 的计算成本

### 一个判断标准

跑一下测试集：看召回的 Top-10 里有多少是"前几个就相关"的。如果 MRR（第一个相关结果的排名倒数均值）已经很高（比如 > 0.8），rerank 的收益有限。如果 MRR 很低（比如 < 0.5），说明排序确实有问题，rerank 能帮上忙。

---

## 小结

Rerank 在 RAG 链路里扮演"精装修"的角色：

- **召回是毛坯房**：快速找到候选，速度快但精度一般
- **Rerank 是精装修**：对候选精细排序，让最好的排到前面

两阶段策略（Bi-encoder 召回 + Cross-encoder 重排）是工业界的事实标准。开源模型里 bge-reranker-large 效果扎实，API 服务里 Cohere 是稳妥选择。

但也要注意：rerank 不是万能的。如果召回本身有硬伤（比如关键文档根本就没被召回），rerank 救不了。rerank 解决的是"排序"问题，不是"召回"问题。

---

## 常见问题

**Q：rerank 会增加多少延迟？**

A：取决于候选数量和模型大小。重排 50 个 doc 约 100ms，100 个约 200ms。用量化或小模型可以降到 50-80ms。

**Q：可以用 LLM 直接 rerank 吗？**

A：可以，用 LLM-as-ranker。效果确实比专门的 reranker 模型好，但成本极高（一个 query 要调用很多次 LLM）。适合离线评估，不适合线上实时用。

**Q：reranker 需要和 embedding 模型配套吗？**

A：不需要。Embedding 模型负责召回，reranker 模型负责重排，两者独立。但最好用同一家的模型（比如都用 BAAI 的），避免打分尺度不一致。

**Q：重排后还需要再重排吗？**

A：一般不需要。两阶段已经足够。极少数场景会用到"多级 rerank"（比如先粗排、再精排、最后再排），但收益通常不大，延迟和成本反而上去了。
