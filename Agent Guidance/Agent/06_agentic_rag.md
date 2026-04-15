# Agentic RAG：让 Agent 主动驾驭检索

上一篇把 RAG 封装成了工具。这一篇讲更进一步的问题：**Agent 如何主动、智能地使用这个工具**，而不只是机械地调用一次。

普通 RAG：用户问 → 检索一次 → 生成答案。

Agentic RAG：用户问 → Agent 分析问题 → 决定检索策略 → 评估结果 → 决定是否再次检索 → 生成答案。

---

## 理论基础：三篇论文

### Self-RAG（2023）

论文：*Self-RAG: Learning to Retrieve, Generate, and Critique*（Asai et al., 2023）

Self-RAG 的核心思想：让模型自己判断什么时候需要检索，以及检索结果是否够好。

它引入了四个特殊 token：

```text
[Retrieve]     — 模型判断是否需要检索
[IsRel]        — 检索到的文档是否与问题相关
[IsSup]        — 生成的内容是否有文档支撑
[IsUse]        — 最终答案是否有用
```

这四个 token 本质上是让模型输出一个"自我评估信号"，驱动检索的决策逻辑。

在工程上不需要训练特殊 token，可以用 LLM 的自然语言判断来模拟这四个信号。

---

### FLARE（2023）

论文：*Active Retrieval Augmented Generation*（Jiang et al., 2023）

FLARE 的思路：**边生成边检索**。

当 LLM 在生成过程中对某些内容不确定时（token 概率低于阈值），停下来检索，用检索结果继续生成。

```text
生成"RAG 系统中 rerank 的..."
    → 继续生成"作用是"
    → 停！下一个词的置信度低
    → 检索"rerank 在 RAG 中的作用"
    → 用检索结果继续生成
```

这个方法对长文档生成特别有效——不是一开始就检索所有内容，而是按需检索。

---

### Adaptive RAG（2024）

论文：*Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models*

Adaptive RAG 的核心：**根据问题复杂度选择不同的检索策略**。

```text
简单问题（单跳）→ 直接用 LLM 内部知识，不检索
中等问题（单次检索）→ 检索一次，生成答案
复杂问题（多跳）→ 多次检索，逐步推理
```

问题复杂度由一个分类器（可以是 LLM）来判断，不需要总是用最重的检索策略。

---

## 工程模式一：Self-RAG 实现

模拟 Self-RAG 的四个判断信号，不需要训练特殊 token：

```python
async def self_rag(query: str) -> str:
    # Step 1：判断是否需要检索
    need_retrieval = await judge_need_retrieval(query)

    if not need_retrieval:
        # 直接用 LLM 知识回答（简单问题）
        return await direct_answer(query)

    # Step 2：执行检索
    retrieval_result = await search_knowledge_base(query)

    # Step 3：判断检索结果是否相关
    is_relevant = await judge_relevance(query, retrieval_result["context"])

    if not is_relevant:
        # 换关键词重试
        new_query = await rewrite_query(query)
        retrieval_result = await search_knowledge_base(new_query)

    # Step 4：生成答案并判断是否有文档支撑
    answer = await generate_with_context(query, retrieval_result["context"])
    is_supported = await judge_support(answer, retrieval_result["context"])

    if not is_supported:
        # 答案超出了检索范围，重新生成更保守的答案
        answer = await generate_conservative(query, retrieval_result["context"])

    return answer


async def judge_need_retrieval(query: str) -> bool:
    prompt = f"""判断以下问题是否需要检索外部知识库才能回答。
如果是闲聊、数学计算、通用常识，回答 NO。
如果涉及产品、技术规格、内部文档，回答 YES。

问题：{query}
答案（只回复 YES 或 NO）："""
    result = await call_llm(prompt)
    return result.strip().upper() == "YES"


async def judge_relevance(query: str, context: str) -> bool:
    prompt = f"""检索到的内容是否与问题相关？

问题：{query}
检索内容（前500字）：{context[:500]}

是否相关（只回复 YES 或 NO）："""
    result = await call_llm(prompt)
    return result.strip().upper() == "YES"


async def judge_support(answer: str, context: str) -> bool:
    prompt = f"""判断答案的核心内容是否有检索结果支撑。
如果答案包含检索结果之外的内容，回答 NO。

检索内容（前500字）：{context[:500]}
答案：{answer}

是否有支撑（只回复 YES 或 NO）："""
    result = await call_llm(prompt)
    return result.strip().upper() == "YES"
```

---

## 工程模式二：多跳检索（Multi-hop）

复杂问题需要多次检索，每次检索的结果决定下一次检索的方向。

场景：
> "我们公司的 RAG 系统用的 rerank 模型，和文档里推荐的最佳实践一致吗？"

这个问题需要两跳：
1. 先检索"我们公司 RAG 系统用什么 rerank 模型"
2. 再检索"RAG rerank 最佳实践"
3. 最后对比两者

```python
async def multi_hop_rag(question: str, max_hops: int = 3) -> str:
    collected_context = []
    current_question = question

    for hop in range(max_hops):
        # 当前跳的检索
        result = await search_knowledge_base(current_question, top_k=5)

        if result["status"] == "no_results":
            break

        collected_context.append(result["context"])

        # 判断是否已经有足够信息回答原问题
        all_context = "\n\n---\n\n".join(collected_context)
        can_answer = await judge_can_answer(question, all_context)

        if can_answer:
            break

        # 生成下一跳的检索问题
        current_question = await generate_next_hop_query(
            original_question=question,
            collected_context=all_context,
            hop_number=hop + 1
        )

    # 基于所有收集到的 context 生成最终答案
    return await generate_final_answer(question, "\n\n---\n\n".join(collected_context))


async def generate_next_hop_query(
    original_question: str,
    collected_context: str,
    hop_number: int
) -> str:
    prompt = f"""原始问题：{original_question}

已检索到的信息（第 1-{hop_number} 跳）：
{collected_context[:1000]}

基于已有信息，还需要检索什么才能完整回答原问题？
请给出下一步的搜索问题（一句话）："""
    return await call_llm(prompt)
```

---

## 工程模式三：查询分解（Query Decomposition）

复杂问题可以先分解成多个子问题，并行检索后汇总：

```python
import asyncio

async def decomposed_rag(question: str) -> str:
    # Step 1：分解问题
    sub_questions = await decompose_question(question)

    # Step 2：并发检索所有子问题
    tasks = [search_knowledge_base(sq, top_k=3) for sq in sub_questions]
    results = await asyncio.gather(*tasks)

    # Step 3：汇总所有检索结果
    combined_context = ""
    for sq, result in zip(sub_questions, results):
        if result["status"] == "success":
            combined_context += f"\n\n### 关于"{sq}"：\n{result['context']}"

    # Step 4：基于汇总 context 回答原问题
    return await generate_with_context(question, combined_context)


async def decompose_question(question: str) -> list[str]:
    prompt = f"""将以下复杂问题分解为 2-4 个独立的子问题，每个子问题可以独立检索回答。
如果问题本身已经简单，只返回原问题本身。

问题：{question}

子问题列表（每行一个）："""
    result = await call_llm(prompt)
    return [q.strip() for q in result.strip().split("\n") if q.strip()]
```

---

## 工程模式四：Adaptive RAG——按复杂度选策略

```python
from enum import Enum

class QueryComplexity(Enum):
    SIMPLE   = "simple"    # 不需要检索，直接回答
    SINGLE   = "single"    # 单次检索
    MULTI    = "multi"     # 多跳检索
    DECOMPOSE = "decompose" # 先分解再检索

async def adaptive_rag(question: str) -> str:
    # 分类问题复杂度
    complexity = await classify_complexity(question)

    if complexity == QueryComplexity.SIMPLE:
        return await direct_answer(question)

    elif complexity == QueryComplexity.SINGLE:
        result = await search_knowledge_base(question)
        return await generate_with_context(question, result["context"])

    elif complexity == QueryComplexity.MULTI:
        return await multi_hop_rag(question)

    elif complexity == QueryComplexity.DECOMPOSE:
        return await decomposed_rag(question)


async def classify_complexity(question: str) -> QueryComplexity:
    prompt = f"""将以下问题分类到对应复杂度级别：

simple   — 闲聊、通用常识、不需要专业知识（如"你好"、"2+2等于几"）
single   — 需要检索一次可以回答（如"rerank 是什么"）
multi    — 需要多步推理，每步的结果决定下一步怎么查（如"A 和 B 相比哪个更适合我的场景"）
decompose — 问题包含多个独立子问题，可以并行检索（如"X 的优点、缺点和适用场景分别是什么"）

问题：{question}
分类（只回复 simple/single/multi/decompose）："""

    result = await call_llm(prompt)
    mapping = {
        "simple": QueryComplexity.SIMPLE,
        "single": QueryComplexity.SINGLE,
        "multi": QueryComplexity.MULTI,
        "decompose": QueryComplexity.DECOMPOSE
    }
    return mapping.get(result.strip().lower(), QueryComplexity.SINGLE)
```

---

## 把 Agentic RAG 集成进 Agent Loop

上面的模式可以作为独立的检索策略层，供 Agent Loop 调用：

```python
# 检索工具：内部使用 Adaptive RAG 策略
async def smart_search(query: str, mode: str = "adaptive") -> dict:
    if mode == "adaptive":
        context = await adaptive_rag(query)
    elif mode == "multi_hop":
        context = await multi_hop_rag(query)
    elif mode == "decompose":
        context = await decomposed_rag(query)
    else:
        result = await search_knowledge_base(query)
        context = result["context"]

    return {"status": "success", "context": context}

# Tool Schema 加上 mode 参数
SMART_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "smart_search",
        "description": "智能知识库检索。支持多种检索模式：adaptive（自动选择）、multi_hop（复杂推理问题）、decompose（多子问题并行检索）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["adaptive", "multi_hop", "decompose", "simple"],
                    "default": "adaptive"
                }
            },
            "required": ["query"]
        }
    }
}
```

---

## 评估 Agentic RAG

Agentic RAG 的评估比普通 RAG 复杂，需要在已有 RAG 评估基础上加上 Agent 维度：

| 评估维度 | 指标 | 工具 |
|---------|------|------|
| 检索质量 | Recall@k, MRR（沿用 RAG 评估）| Ragas |
| 检索决策 | 该检索时检索了吗，不该检索时没检索吗 | 自定义 |
| 多跳准确率 | 多跳问题的最终答案正确率 | 自定义 |
| 幻觉率 | 答案有多少内容超出了检索结果 | Ragas Faithfulness |
| 效率 | 平均检索次数，超出必要检索次数的比例 | 自定义 |

```python
# 检索决策准确率：标注哪些问题需要检索，哪些不需要
def eval_retrieval_decision(test_cases: list[dict]) -> float:
    correct = 0
    for case in test_cases:
        predicted = run_need_retrieval_judge(case["question"])
        if predicted == case["needs_retrieval"]:
            correct += 1
    return correct / len(test_cases)
```

---

## 工程现场

场景：多跳 RAG 在第二跳偏离了原始问题，越跑越远，最终给出了一个和原问题毫无关系的答案。

原因：`generate_next_hop_query` 在生成下一个子问题时，没有足够强调"必须服务于原始问题"。

修复：在生成下一跳 query 的 prompt 里，每次都把原始问题再次强调：

```python
prompt = f"""原始问题（始终要服务于这个）：{original_question}

已有信息揭示了什么？
{collected_context[:800]}

还缺什么信息才能回答原始问题？给出一个精准的检索问题："""
```

同时加一个判断：如果下一跳的 query 和原始问题语义距离超过阈值，强制中止多跳，基于已有信息直接回答。

---

## 小结

Agentic RAG 的核心不是"更多检索"，而是"更智能的检索决策"：

```text
Self-RAG    — 自我评估：检索结果够不够、答案有没有依据
Multi-hop   — 逐步推理：每步结果决定下一步怎么检索
Decompose   — 并行检索：把问题拆成子问题同时检索
Adaptive    — 按需选策略：根据问题复杂度匹配检索深度
```

这四个模式不是互斥的，可以组合使用。生产建议：先上 Adaptive 策略（能覆盖大多数场景），再根据评估数据发现的问题补充 Self-RAG 和 Multi-hop。

阶段二到这里结束。下一步进入阶段三：LangGraph——用状态图来构建复杂 Agent。
