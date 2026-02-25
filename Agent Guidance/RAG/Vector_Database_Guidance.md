
Embedding

Similarity 
* L2 Distance
* Dot Product
* Cosine Distance

检索算法
* ANN -> KNN -> Faiss
* IVF 
* HNSW
* PQ
* LSH、PQ、ISV-PQ

''''
Embedding 理论
    ↓
相似度度量
    ↓
ANN 算法
    ↓
索引结构优化
    ↓
Chunk 设计
    ↓
Hybrid 检索
    ↓
Rerank
    ↓
Prompt 组装
    ↓
评估体系
    ↓
Agent 动态检索
    ↓
生产部署
''''


Temp_Guidance
很好，这个问题已经进入**真正的高阶工程层**了。

我给你两部分内容：

1️⃣ 面试级 RAG 知识清单（能打大厂）  
2️⃣ 生产级 RAG 架构图（能落地 10M+ 文档）

我会按「面试官会怎么问」来组织。

---

# 第一部分：面试级 RAG 知识清单

---

# 一、Embedding & 相似度（必须能讲透）

### 1️⃣ Cosine / Dot / L2 关系

genui{"math_block_widget_always_prefetched": {"content": "cos(\theta)=\frac{A\cdot B}{||A||\,||B||}"}}

如果向量归一化：

genui{"math_block_widget_always_prefetched": {"content": "||A-B||^2 = 2 - 2\cos\theta"}}

你必须能回答：

- 为什么 embedding 通常要 normalize？
    
- 为什么 Faiss 默认用 L2 也能等价 cosine？
    
- MIPS 如何转成 L2？
    

---

# 二、ANN 原理（不只是会用 Faiss）

必须掌握：

### 1️⃣ IVF

- coarse quantizer
    
- nlist
    
- nprobe
    
- recall vs latency tradeoff
    

### 2️⃣ HNSW

- graph-based search
    
- efConstruction
    
- efSearch
    
- 内存换时间
    

### 3️⃣ PQ

- 子空间量化
    
- 8bit code
    
- 内存压缩原理
    

### 4️⃣ Hybrid Index

- IVF + PQ
    
- HNSW + PQ
    

面试会问：

> 1000万文档，如何控制 50ms 内响应？

必须能讲：

- 分片
    
- 多副本
    
- 召回+rerank
    

---

# 三、RAG Pipeline 设计（核心）

你必须能完整讲一条流水线：

```text
用户 Query
   ↓
Query Rewrite
   ↓
Hybrid Retrieval
   ↓
Rerank
   ↓
Context Assemble
   ↓
LLM Generate
   ↓
Post-process
```

---

# 四、Chunk 策略（90%候选人挂在这里）

必须掌握：

- 固定长度切分
    
- 滑动窗口
    
- 语义切分
    
- 父子块索引（Parent-Child）
    

面试经典问题：

> chunk 大小如何选？

回答要点：

- 太小 → 语义丢失
    
- 太大 → recall 下降
    
- 取决于 embedding receptive field
    

---

# 五、Hybrid 检索

现代 RAG 标配：

```text
BM25 + Dense Retrieval
```

必须会讲：

- keyword 精准匹配优势
    
- dense 语义匹配优势
    
- score fusion
    

---

# 六、Reranker（高频考点）

必须了解：

- Cross-encoder
    
- bi-encoder 区别
    
- 为什么 rerank 能提升 precision
    

---

# 七、RAG 评估体系（很多人不会）

必须掌握：

- Recall@k
    
- MRR
    
- NDCG
    
- Faithfulness
    
- LLM-as-judge
    

面试问：

> 如何评估 RAG 是否优化成功？

不能回答“感觉更准”。

---

# 八、Agent 级 RAG（加分项）

必须能讲：

- Self-RAG
    
- 动态是否检索
    
- 多轮检索
    
- memory 系统
    

---

# 九、生产问题（决定你级别）

必须能回答：

- 索引如何在线更新？
    
- embedding 如何批量生成？
    
- 冷热数据分层？
    
- 如何做 A/B test？
    

---

# 第二部分：生产级 RAG 架构图

假设：

- 1000万文档
    
- 50 QPS
    
- 200ms SLA
    
- 支持实时更新
    

---

# 架构总览

```text
                ┌──────────────┐
                │   API Layer  │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │ Query Router │
                └──────┬───────┘
                       ↓
          ┌───────────────────────────┐
          │   Retrieval Layer         │
          │  ─ Hybrid Search          │
          │  ─ Vector DB (IVF+PQ)     │
          │  ─ BM25 (ES)              │
          └───────────┬───────────────┘
                      ↓
              ┌───────────────┐
              │  Reranker     │
              └──────┬────────┘
                     ↓
              ┌───────────────┐
              │ Context Builder│
              └──────┬────────┘
                     ↓
              ┌───────────────┐
              │ LLM Gateway   │
              └──────┬────────┘
                     ↓
              ┌───────────────┐
              │  Response     │
              └───────────────┘
```

---

# 数据层设计

## 1️⃣ 向量库

推荐：

- IVF-PQ 或 HNSW
    
- 分 shard
    
- 按业务分 collection
    

## 2️⃣ 关键词检索

- Elasticsearch
    
- OpenSearch
    

---

# 更新流程

```text
新文档
   ↓
ETL 清洗
   ↓
Chunk
   ↓
Embedding 批量生成
   ↓
写入 Vector DB + ES
```

必须支持：

- 异步 pipeline
    
- 消息队列
    
- 批量更新
    

---

# 性能优化策略

## 延迟优化

- nprobe 控制
    
- top-k 控制
    
- rerank 数量控制
    
- embedding cache
    

## 成本优化

- 小模型 embedding
    
- LLM 分级路由
    
- 批量请求
    

---

# 10M 规模关键点

- PQ 压缩
    
- 分片并行查询
    
- GPU 检索（Faiss GPU）
    
- 读写分离
    

---

# 真正的高阶思考

面试官更喜欢问：

> 长上下文模型是否会取代 RAG？

回答：

- 长上下文解决的是输入容量
    
- RAG 解决的是知识更新、可控性、可审计性
    
- 企业场景 RAG 不会消失
    

---

# 总结：你现在的级别差在哪？

你已经具备：

✔ 向量检索算法基础

你需要补：

✔ 评估体系  
✔ Hybrid + Rerank  
✔ 系统工程  
✔ 更新策略  
✔ 性能调优

---

如果你愿意，我可以下一步带你做：

1️⃣ 设计一个「1000万文档真实企业知识库」  
2️⃣ 或模拟一次“大厂RAG面试深挖”  
3️⃣ 或带你从0实现一个生产级 demo 架构

你想往哪一个方向打磨？