# 07. Query Enhancement（查询增强）

## 1. 章内目标

通过改写、拆解或生成代理查询，提升召回覆盖率与稳定性。

---

## 2. 方法与适用场景

| 方法      | 适用问题             | 成本         |
| --------- | -------------------- | ------------ |
| Rewriting | query 过短、表达含糊 | 低           |
| Step-back | 机制类问题、背景不足 | 低           |
| Sub-query | 多约束复合问题       | 中           |
| HyDE      | query-doc 语体差异大 | 中高         |
| HyPE      | 高并发、语料稳定     | 高（索引期） |

---

## 3. 工程策略

1. 先上 Rewriting，再加 Sub-query。
2. HyDE 与原 query 并行召回，避免单路失败。
3. HyPE 只在稳定语料和高流量场景启用。
4. 所有扩展 query 要做去重与长度上限。

---

## 4. 工程现场

场景：复杂问题“法律+税务+时效”召回不稳。

改造：

1. Sub-query 拆成 3 路检索。
2. 每路 top-k 限 20。
3. 融合后去重再 rerank。

结果：Recall 提升，但 P95 上升；后续通过缓存 query 转换结果回收延迟。

---

## 5. 面试问答

Q1：什么时候不该用 HyDE？

A：延迟预算极紧或对生成偏差容忍度很低时，不建议默认启用。

Q2：Sub-query 的核心风险是什么？

A：query 扩张导致候选爆炸和重复证据，需要强去重和预算约束。

Q3：HyPE 和 HyDE 的本质区别？

A：HyDE 在查询时生成假设文档；HyPE 在索引时生成假设问题，把成本前置。

---

## 6. 参考

- Query Transformations: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/query_transformations.ipynb>
- HyPE: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/HyPE_Hypothetical_Prompt_Embeddings.ipynb>
- HyDE: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb>
