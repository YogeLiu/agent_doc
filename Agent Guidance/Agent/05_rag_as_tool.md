# RAG as Tool：把检索链路封装为 Agent 工具

你已经有一条完整的 RAG 链路：Query Enhancement → 多路召回 → Fusion → Rerank → Context Builder → Generation。

现在要做的，不是重建这条链路，而是把它**封装成 Agent 的一个工具**。

这个过程本质上是：把 RAG 的"同步管道"改造成 Agent 可以按需调用的"工具接口"。

---

## 为什么要封装成工具而不是内嵌在 Agent 里

直接把 RAG 逻辑写死在 Agent Loop 里也能跑，但会产生两个问题：

**问题一：每次推理都触发检索，不管需不需要**

Agent 处理"你好"这种问候时，完全不需要检索，但如果 RAG 是内嵌的，仍然会触发一次向量搜索。

**问题二：检索策略无法动态调整**

有的问题需要深度检索（top-k=20，开启 rerank），有的只需要快速粗检索（top-k=5，跳过 rerank）。封装成工具后，Agent 可以根据问题性质传不同参数。

封装成工具之后，**LLM 自己决定要不要检索、检索什么、用什么参数**。

---

## 最简封装：单一检索工具

把 RAG 链路最核心的入口（query → context chunks）封装成一个工具：

```python
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    rerank: bool = True
) -> dict:
    """
    封装已有 RAG 链路的核心逻辑
    """
    # 1. Query Enhancement（可选，沿用已有实现）
    enhanced_queries = await query_enhancement(query)

    # 2. 多路召回（沿用已有实现）
    dense_results  = await dense_search(query, top_k=top_k * 3)
    sparse_results = await sparse_search(query, top_k=top_k * 3)
    expanded_results = []
    for eq in enhanced_queries:
        expanded_results += await dense_search(eq, top_k=20)

    # 3. Fusion（RRF，沿用已有实现）
    fused = rrf_fusion([dense_results, sparse_results, expanded_results])

    # 4. Rerank（按参数决定是否启用）
    if rerank and len(fused) > top_k:
        fused = await rerank_results(query, fused, top_k=top_k)
    else:
        fused = fused[:top_k]

    # 5. Context Builder：把 chunks 格式化为 Agent 可用的字符串
    context = build_context(fused)

    return {
        "status": "success",
        "query": query,
        "chunks_found": len(fused),
        "context": context,          # 给 LLM 读的文本
        "sources": [c["source"] for c in fused]  # 来源列表，用于引用
    }

# 对应的 Tool Schema
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "在知识库中检索相关信息。适用于回答关于产品功能、技术文档、内部规范的问题。对于闲聊、数学计算等不需要知识支撑的问题，无需调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题，用你自己的语言重新表述，不要直接复制用户原话"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的文档片段数量。简单问题填 3-5，复杂问题填 8-10",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "rerank": {
                    "type": "boolean",
                    "description": "是否启用精排。需要高精度时设为 true，需要快速响应时设为 false",
                    "default": True
                }
            },
            "required": ["query"]
        }
    }
}
```

---

## 拆分多个专用检索工具

单一工具够用，但在知识库有多个领域时，拆分成多个工具效果更好。

原因：LLM 会根据工具 description 选择工具。描述越精准，LLM 选对工具的概率越高。

```python
tools = [
    {
        "name": "search_product_docs",
        "description": "搜索产品功能文档、操作手册、发版记录。用于回答'如何使用 X 功能'、'X 功能在哪里'类问题。",
        # ...
    },
    {
        "name": "search_tech_specs",
        "description": "搜索技术规格、API 文档、系统架构设计文档。用于回答接口参数、性能指标、技术实现类问题。",
        # ...
    },
    {
        "name": "search_support_cases",
        "description": "搜索历史支持案例和解决方案。用于回答'遇到 X 报错怎么解决'、'X 问题有没有解决方案'类问题。",
        # ...
    }
]
```

---

## Context Builder：RAG 输出如何给 LLM 读

这是容易被忽视但很关键的一步。检索到了 chunks，怎么格式化给 LLM？

**方案一：直接拼接（简单场景）**

```python
def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "未知来源")
        content = chunk["content"]
        parts.append(f"[{i}] 来源：{source}\n{content}")
    return "\n\n---\n\n".join(parts)
```

**方案二：带结构的 XML 格式（复杂场景，效果更好）**

```python
def build_context_xml(chunks: list[dict]) -> str:
    parts = ["<retrieved_context>"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"""<document index="{i}">
<source>{chunk.get('source', '未知')}</source>
<relevance_score>{chunk.get('score', 0):.2f}</relevance_score>
<content>{chunk['content']}</content>
</document>""")
    parts.append("</retrieved_context>")
    return "\n".join(parts)
```

XML 格式让 LLM 更容易区分不同 chunk 的边界，在 chunk 数量多时效果明显更好。

**System Prompt 里需要告诉 LLM 如何使用检索结果**：

```python
SYSTEM_PROMPT = """你是一个知识库问答助手。

回答规则：
1. 必须基于 search_knowledge_base 返回的内容回答，不要使用你的训练知识
2. 回答时引用来源，格式：（来源：[文档名]）
3. 如果检索结果不足以回答，说明"当前知识库中没有找到相关信息"，不要猜测
4. 如果需要更多信息，可以用不同关键词再次搜索
"""
```

---

## 工具返回值的设计

工具返回值直接影响 LLM 的后续推理质量。

**不好的返回值：**

```python
# 只返回文本，LLM 没有结构信息
return "找到了一些关于 rerank 的内容：rerank 是用来提升检索精度的..."
```

**好的返回值：**

```python
return {
    "status": "success",          # 明确的状态信号
    "chunks_found": 3,            # 找到几条结果
    "context": "...",             # 具体内容
    "sources": ["doc_a", "doc_b"], # 来源列表
    "search_query": query         # 实际搜索的 query（方便调试）
}
```

**没有结果时的返回值：**

```python
# 没有结果时不要返回空，要给 LLM 明确的信号
if not chunks:
    return {
        "status": "no_results",
        "message": f"没有找到与'{query}'相关的文档。建议尝试更换关键词或扩大搜索范围。",
        "chunks_found": 0
    }
```

LLM 看到 `status: no_results` 后，通常会主动换关键词重新搜索，而不是基于空内容瞎编答案。

---

## 多次检索的处理

Agent 在一次任务里可能多次调用 `search_knowledge_base`，需要对结果去重：

```python
class RAGToolWithDedup:
    def __init__(self):
        self._seen_chunk_ids = set()

    async def search(self, query: str, top_k: int = 5) -> dict:
        raw_results = await search_knowledge_base_raw(query, top_k=top_k * 2)

        # 去掉本次任务里已经见过的 chunk
        new_results = [
            r for r in raw_results
            if r["id"] not in self._seen_chunk_ids
        ]
        for r in new_results[:top_k]:
            self._seen_chunk_ids.add(r["id"])

        return build_response(new_results[:top_k])

    def reset(self):
        """每次新任务开始时重置"""
        self._seen_chunk_ids.clear()
```

---

## 引用追踪

生产环境通常需要知道答案来自哪些文档。在工具返回里加 source 信息，让 LLM 在回答时引用：

```python
# System Prompt 里的引用规则
"回答中必须标注引用来源，格式：[文档名, 第X段]"

# 工具返回里带上来源信息
return {
    "context": formatted_context,
    "citations": [
        {"index": 1, "source": "RAG技术文档.pdf", "page": 3},
        {"index": 2, "source": "系统架构设计.md", "section": "检索层"}
    ]
}
```

---

## 工程现场

场景：RAG 工具被调用后，LLM 生成的答案包含了检索结果之外的信息（幻觉），用户反馈答案不准确。

根本原因：System Prompt 没有明确约束 LLM 只能基于检索结果回答。

修复：

```python
SYSTEM_PROMPT = """...

重要约束：
- 你只能基于 search_knowledge_base 返回的内容来回答问题
- 如果检索结果中没有相关信息，回复"知识库中没有找到相关信息"
- 禁止使用你自己的训练知识补充答案，即使你认为你知道
"""
```

同时在评估阶段用 Faithfulness 指标（来自 Ragas）监控答案是否忠实于检索结果，及时发现问题。

---

## 小结

RAG as Tool 的核心改造点：

```text
原来的 RAG 管道（同步、必然触发）：
    Query → Enhancement → Retrieval → Rerank → Generation

封装成工具后（按需、可控）：
    Agent 决定是否检索
        ↓（需要时）
    search_knowledge_base(query, top_k, rerank)
        ↓
    返回结构化结果 → 追加到 messages → LLM 继续推理
```

改造工作量不大，但带来的收益是：检索变成了 Agent 的一个可控行动，而不是一个必经管道。

下一篇讲 Agentic RAG：Agent 不只是被动调用 RAG，而是主动决定何时检索、检索什么、以及对检索结果做自我评估。
