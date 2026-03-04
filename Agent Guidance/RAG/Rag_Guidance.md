# RAG 技术博客：一条能上线的检索增强链路

## 1. 先定目标，不先选模型

RAG 项目成败通常不在模型，而在链路：

1. 能不能稳定找回关键证据。
2. 能不能基于证据回答，而不是“像在回答”。
3. 延迟和成本能不能在 SLA 内长期运行。

---

## 2. 一条紧凑主链路

```text
Query
 -> Query Enhancement (optional)
 -> Multi-path Retrieval
 -> Fusion
 -> Rerank
 -> Context Enhancement (optional)
 -> Generation
 -> Evaluate + Monitoring
```

---

## 3. 检索层关键：多路召回 + metadata 前置过滤

推荐并行四路：

1. Dense（语义）
2. Sparse/BM25（实体与数字）
3. Expanded Query（复杂问题拆解）
4. Rule/Filter（租户、权限、时间约束）

融合建议先用 RRF：

```text
RRF(Dense, Sparse, Expanded, RuleFiltered)
 -> dedup
 -> rerank
```

起步参数：

- dense top-k: 50
- sparse top-k: 50
- expanded 每路 top-k: 20~30
- 融合后候选: 80~150

为什么 filter 要前置：

- 后置过滤会浪费 rerank 预算。
- 会把跨域噪声带到生成阶段。

---

## 4. Rerank 与 Context 的配合

1. Rerank 负责排序正确。
2. Context Enhancement 负责证据组织完整。

顺序建议：先把 rerank 做稳，再上 context 增强。

---

## 5. 评估闭环

每次改动至少看四类指标：

1. 检索：Recall@k, MRR, nDCG。
2. 生成：Faithfulness, Answer Relevance。
3. 系统：P95, QPS, 成本/千问。
4. 线上：空结果率、重试率、人工接管率。

---

## 6. 工程现场

场景：离线指标持续上涨，线上用户却说“更慢更不准”。

排查后常见是三件事叠加：

1. query 扩展后候选数暴涨。
2. metadata 过滤后置，导致无效候选进入重排。
3. context 拼接没有去重，重复证据挤占窗口。

改造顺序：

1. 过滤前置到各召回路。
2. 融合后先去重再 rerank。
3. 对扩展 query 设 hard budget。

---

## 7. 面试问答（总览）

Q1：为什么不是 dense-only？

A：dense 语义强，但对实体、数字、版本号稳定性不足。生产里通常要 sparse 兜底。

Q2：RRF 和加权融合怎么选？

A：早期选 RRF 稳定上线；中后期有评测体系后再做加权精调。

Q3：metadata 过滤放在召回前还是召回后？

A：召回前。后置会浪费重排资源并增加误答。

---

## 8. 文件组织（按标题阅读）

- `01_embedding_model.md`
- `02_vector_search.md`
- `03_sparse_retrieval.md`
- `04_hybrid_search.md`
- `05_rerank.md`
- `06_evaluate.md`
- `07_query_enhancement.md`
- `08_context_enhancement.md`
- `chunking.md`
- `vector_database.md`
