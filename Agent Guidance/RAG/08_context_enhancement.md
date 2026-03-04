# 08. Context Enhancement（上下文增强）

## 1. 章内目标

提升“送给 LLM 的证据质量”，解决上下文断裂、冗余和噪声问题。

---

## 2. 方法选择

| 方法                   | 解决问题             | 成本         |
| ---------------------- | -------------------- | ------------ |
| CCH                    | 孤立 chunk 缺背景    | 索引重建     |
| RSE                    | 命中离散，缺连续证据 | 中           |
| CEW                    | 快速补前后文         | 低           |
| Contextual Compression | 长候选噪声高         | 中高         |
| Document Augmentation  | 问法多样导致漏召     | 高（索引期） |

---

## 3. 组合顺序

1. 快速见效：CEW。
2. 连贯性优先：CCH + RSE。
3. 成本治理：再加 Compression。
4. 召回上限不足：最后引入 Document Augmentation。

---

## 4. 工程现场

场景：检索结果看起来相关，但回答经常“缺半段结论”。

原因：证据跨 chunk，且 chunk 没有重建连续上下文。

处理：

1. 命中后先做 CEW(`num_neighbors=1`)。
2. 对财报/条款场景切到 RSE。
3. 对超长上下文再启用压缩。

---

## 5. 面试问答

Q1：CEW 和 RSE 怎么选？

A：CEW 实现快、成本低；RSE 适合必须恢复连续段落证据的场景，效果更稳。

Q2：Contextual Compression 的风险是什么？

A：压缩过头会丢关键句，需要保留原文映射和回退机制。

Q3：为什么 context 增强不能替代 rerank？

A：两者职责不同。rerank 是排序，context 增强是证据组织。

---

## 6. 参考

- CCH: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb>
- RSE: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/relevant_segment_extraction.ipynb>
- CEW: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/context_enrichment_window_around_chunk.ipynb>
- Contextual Compression: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/contextual_compression.ipynb>
- Document Augmentation: <https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/document_augmentation.ipynb>
