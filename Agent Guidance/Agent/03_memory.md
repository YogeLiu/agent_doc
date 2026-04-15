# Memory：Agent 的记忆系统

Agent Loop 里的 messages 列表，是最基础的记忆形式。但只有 messages 是不够的：

- messages 有 context window 限制，超长后必须截断
- messages 在会话结束后消失，下次对话从零开始
- 复杂任务里，当前步骤的执行状态需要独立管理

这就是 Agent 需要完整记忆系统的原因。

---

## 理论基础：三篇重要论文

在写代码之前，先理解研究方向。这三篇论文建立了 Agent 记忆系统的核心框架。

### CoALA（2023）——记忆分类的理论基础

论文：*Cognitive Architectures for Language Agents*（Sumers et al., 2023）

CoALA 借鉴认知科学，把 Agent 的记忆系统分为四类：

```text
Working Memory   — 当前上下文，即 context window 内的信息
Episodic Memory  — 过去的经历（具体对话、任务执行历史）
Semantic Memory  — 通用知识（事实、规则、领域知识）
Procedural Memory — 操作知识（如何做某件事，如技能、代码库）
```

这个分类比简单的"短期/长期"更精确。它告诉我们：向量数据库适合存 Episodic + Semantic，而 Procedural 更适合存为代码或结构化规则。

---

### Generative Agents（2023）——记忆的完整工程实现

论文：*Generative Agents: Interactive Simulacra of Human Behavior*（Park et al., 2023，Stanford + Google）

这篇论文造了 25 个 AI "角色"生活在一个模拟小镇里，每个角色有完整的记忆系统。它提出了三个至今仍被广泛采用的机制：

**机制一：Memory Stream（记忆流）**

每次感知到的信息、执行的动作、产生的想法，都按时间顺序追加到记忆流里。记忆流是所有记忆的原始来源。

**机制二：三维检索打分**

检索时不只看语义相似度，而是三个维度加权：

```text
score = α · recency + β · importance + γ · relevance

recency   — 近期发生的记忆权重更高（指数衰减）
importance — 记忆的重要程度（由 LLM 对每条记忆打 1-10 分）
relevance  — 与当前查询的语义相似度
```

这个公式解决了"语义相似但已过时的记忆"和"最近发生但语义不太相关的记忆"的权衡问题。

**机制三：Reflection（反思）**

当记忆积累到一定量后，Agent 会自动触发 Reflection：

```text
从最近的记忆里，提炼出 3-5 条高层洞察
把这些洞察作为新的记忆存入记忆流
```

Reflection 让 Agent 能从具体经历里提炼规律，而不只是记住事件本身。

---

### MemGPT（2023）——分层记忆管理

论文：*MemGPT: Towards LLMs as Operating Systems*（Packer et al., 2023，UC Berkeley）

MemGPT 的核心类比：把 LLM 类比成 CPU，context window 类比成内存（RAM），外部存储类比成磁盘。

```text
Main Context（RAM）
    ├── System Prompt（常驻，不换出）
    ├── Working Memory（频繁访问的信息）
    └── FIFO Queue（最近的对话历史）

External Storage（磁盘）
    ├── Archival Memory（归档的长期记忆）
    └── Recall Memory（可检索的历史对话）
```

关键思想：Agent 本身可以主动管理自己的记忆——决定什么时候把 context 里的信息"换出"到外部存储，什么时候从外部存储"换入"相关信息。

MemGPT 让 Agent 拥有了两个特殊工具：`memory_search`（从外部存储检索）和 `memory_append`（把信息存入外部存储）。

---

### A-MEM（2024）——动态记忆组织

论文：*A-MEM: Agentic Memory for LLM Agents*（2024）

A-MEM 的核心思想：记忆不应该是静态存储的，而应该像人类大脑一样动态组织。

每次存入新记忆时：
1. 为这条记忆生成关键词、上下文描述、链接到相关已有记忆
2. 已有记忆也会被更新——新记忆可能改变旧记忆的重要性或关联关系

这形成了一个动态演化的记忆网络，而不是静态的向量库。

---

## 三种记忆类型（工程实现）

```text
Short-term Memory（短期记忆）
    当前对话的 messages 历史
    生命周期：一次会话

Working Memory（工作记忆）
    当前任务的执行状态（计划、已完成步骤、中间结果）
    生命周期：一次任务执行

Long-term Memory（长期记忆）
    跨会话的持久信息（用户偏好、历史交互、领域知识）
    生命周期：持久存储
```

---

## Short-term Memory：对话历史管理

### 超长截断

```python
import tiktoken

def count_tokens(messages: list, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += len(enc.encode(str(msg.get("content", ""))))
    return total

def trim_to_budget(
    messages: list,
    max_tokens: int = 100000,
    model: str = "gpt-4o"
) -> list:
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    kept = []
    token_count = count_tokens(system_msgs, model)

    for msg in reversed(other_msgs):
        msg_tokens = count_tokens([msg], model)
        if token_count + msg_tokens > max_tokens:
            break
        kept.insert(0, msg)
        token_count += msg_tokens

    # 保证 tool call 和 tool result 配对完整
    while kept and kept[0].get("role") == "tool":
        kept = kept[1:]

    return system_msgs + kept
```

### 历史摘要

对话很长时，不截断而是压缩，保留语义：

```python
async def summarize_old_messages(messages: list, keep_recent: int = 10) -> list:
    if len(messages) <= keep_recent:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    to_summarize = other_msgs[:-keep_recent]
    recent = other_msgs[-keep_recent:]

    summary = await call_llm(f"""将以下对话历史压缩为简洁摘要，保留关键信息和结论：

{format_messages(to_summarize)}""")

    return system_msgs + [
        {"role": "system", "content": f"[对话历史摘要] {summary}"}
    ] + recent
```

---

## Working Memory：任务状态管理

Working Memory 存储当前任务的执行状态。LangGraph 的 State 就是 Working Memory 的框架实现。

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TaskState:
    task: str = ""
    plan: list[str] = field(default_factory=list)
    completed_steps: list[dict] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    step_count: int = 0
    token_used: int = 0

async def run_agent_with_state(task: str) -> str:
    state = TaskState(task=task)
    messages = [{"role": "user", "content": task}]

    while state.step_count < MAX_STEPS:
        response = await call_llm(messages, tools)
        message = response.choices[0].message

        state.step_count += 1
        state.token_used += response.usage.total_tokens

        if not message.tool_calls:
            state.final_answer = message.content
            return state.final_answer

        for tool_call in message.tool_calls:
            result = await execute_tool(tool_call)
            state.tool_results[tool_call.function.name] = result
            state.completed_steps.append({
                "step": state.step_count,
                "tool": tool_call.function.name,
                "result_summary": str(result)[:200]
            })

        # 每 5 步注入一次状态摘要，防止 LLM 遗忘
        if state.step_count % 5 == 0:
            messages.append({
                "role": "system",
                "content": f"当前进展（第 {state.step_count} 步）：{summarize_state(state)}"
            })

    return state.final_answer or "任务未完成"
```

---

## Long-term Memory：跨会话持久记忆

基于 Generative Agents 的三维打分思路，实现一个带重要性评分的长期记忆：

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import math, uuid

class LongTermMemory:
    def __init__(self, client: QdrantClient, collection: str = "agent_memory"):
        self.client = client
        self.collection = collection

    async def store(self, content: str, metadata: dict = None):
        # 让 LLM 对这条记忆打重要性分（1-10）
        importance = await self._score_importance(content)

        embedding = await get_embedding(content)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": content,
                    "importance": importance,
                    "timestamp": datetime.now().timestamp(),
                    "access_count": 0,
                    **(metadata or {})
                }
            )]
        )

    async def recall(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = await get_embedding(query)
        now = datetime.now().timestamp()

        # 检索候选（多取一些，后面用三维打分重排）
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k * 3,
            with_payload=True
        )

        # 三维打分（参考 Generative Agents）
        scored = []
        for r in results:
            payload = r.payload
            hours_ago = (now - payload["timestamp"]) / 3600
            recency = math.exp(-0.01 * hours_ago)           # 指数衰减
            importance = payload["importance"] / 10.0        # 归一化到 0-1
            relevance = r.score                              # 向量相似度

            final_score = 0.3 * recency + 0.3 * importance + 0.4 * relevance
            scored.append({"content": payload["content"], "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def _score_importance(self, content: str) -> float:
        prompt = f"""对以下信息的重要性打分（1-10分），1分=琐碎，10分=非常重要。
只回复数字。

信息：{content}
分数："""
        result = await call_llm(prompt)
        try:
            return float(result.strip())
        except:
            return 5.0
```

### 整合到 Agent Loop

```python
async def run_agent_with_longterm_memory(
    user_input: str,
    user_id: str,
    memory: LongTermMemory
):
    # 检索相关历史记忆
    relevant_memories = await memory.recall(user_input, top_k=3)
    memory_context = "\n".join([f"- {m['content']}" for m in relevant_memories])

    messages = [
        {
            "role": "system",
            "content": f"你是一个助手。\n\n{'用户历史信息：\n' + memory_context if memory_context else ''}"
        },
        {"role": "user", "content": user_input}
    ]

    result = await run_agent(messages)

    # 会话结束后，提炼关键信息存入长期记忆
    key_info = await extract_key_info(user_input, result)
    if key_info:
        await memory.store(key_info, metadata={"user_id": user_id})

    return result

async def extract_key_info(user_input: str, response: str) -> str | None:
    prompt = f"""分析以下对话，判断是否有值得长期记住的信息（用户偏好、重要事实、关键结论）。
如果有，提炼成一句话。如果没有，回复 NONE。

用户：{user_input}
助手：{response}

重要信息："""
    result = await call_llm(prompt)
    return None if result.strip() == "NONE" else result.strip()
```

---

## MemGPT 启发的记忆换入换出

对于超长任务，可以让 Agent 自己管理记忆的换入换出，给它两个特殊工具：

```python
memory_tools = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "在长期记忆中搜索信息。当你需要回忆过去的对话或已知事实时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "把重要信息存入长期记忆。当你判断某个信息值得记住时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要记住的信息"}
                },
                "required": ["content"]
            }
        }
    }
]
```

Agent 在运行时可以主动调用这两个工具，自主决定何时存入、何时检索记忆。这是 MemGPT 的核心思想——记忆管理本身也是 Agent 的一种行动能力。

---

## 三种记忆的配合

```text
用户发送新消息
    │
    ├── Long-term Memory 检索相关记忆 → 注入 System Prompt
    ├── ConversationStore 加载 Short-term Memory（历史 messages）
    └── 初始化 Working Memory（TaskState）
    │
    ▼
Agent Loop 运行
    │
    ├── Short-term Memory：messages 持续追加
    ├── Working Memory：TaskState 持续更新
    └── 超长时：Short-term Memory 做摘要压缩
    │
    ▼
任务完成
    │
    ├── Short-term Memory → 保存到 ConversationStore
    └── 提炼关键信息 → 存入 Long-term Memory
```

---

## 工程现场

场景：Agent 在 10 步任务中，步骤 7 完全忘记了步骤 2 找到的关键信息，重新搜索了一遍。

原因：messages 太长，步骤 2 的结果被挤到 context 尾部，LLM 注意力下降后忽略了它。

修复：引入 Working Memory，把每步的关键结果单独存在 `task_state.tool_results` 里，并在每隔几步时注入一次状态摘要到 messages 中，强制 LLM 看到已有的信息。

---

## 小结

| 记忆类型 | 论文来源 | 存在哪里 | 生命周期 | 核心解决什么 |
|---------|---------|---------|---------|------------|
| Short-term | — | messages 列表 | 单次会话 | 对话连贯性 |
| Working | CoALA | TaskState 对象 | 单次任务 | 多步任务状态追踪 |
| Long-term | Generative Agents, MemGPT | 向量数据库 | 持久 | 跨会话个性化与知识积累 |
| 自主记忆管理 | MemGPT, A-MEM | 外部存储 + 工具 | 持久 | Agent 自主决定记忆读写 |

大多数生产 Agent 至少需要前两种。Long-term Memory 在对话机器人、个人助手类场景里价值最大。

下一篇讲 Streaming：如何让 Agent 的多步执行过程实时输出给用户，而不是等所有步骤完成后才响应。
