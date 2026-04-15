# LangGraph 进阶：并发、子图、Human-in-the-loop

基础篇建立了 State / Node / Edge 三件套。进阶篇解决生产 Agent 必须面对的四个问题：

1. 多个独立任务怎么并发执行
2. 复杂图怎么拆成子图复用
3. 关键节点怎么让人审批后再继续
4. Agent 执行中途崩溃了怎么恢复

---

## 并发节点（Parallel Execution）

有些步骤可以同时执行——比如同时查知识库和查数据库，不需要串行等待。

LangGraph 的并发方式：**多条 Edge 从同一个 Node 指向多个 Node**。

```python
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from operator import add


class State(TypedDict):
    messages: Annotated[list, add_messages]
    kb_result: str        # 知识库检索结果
    db_result: str        # 数据库查询结果
    final_answer: str


# 两个独立的检索 Node
def search_kb(state: State) -> dict:
    result = do_kb_search(state["messages"][-1].content)
    return {"kb_result": result}

def query_db(state: State) -> dict:
    result = do_db_query(state["messages"][-1].content)
    return {"db_result": result}

# 汇总 Node
def synthesize(state: State) -> dict:
    answer = combine(state["kb_result"], state["db_result"])
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


# 构建图：search_kb 和 query_db 并发执行
graph = StateGraph(State)
graph.add_node("search_kb", search_kb)
graph.add_node("query_db", query_db)
graph.add_node("synthesize", synthesize)

# 入口同时指向两个 Node → 并发
graph.set_entry_point("search_kb")
graph.set_entry_point("query_db")  # 多入口表示并发

# 两个 Node 都完成后，汇合到 synthesize
graph.add_edge("search_kb", "synthesize")
graph.add_edge("query_db", "synthesize")
graph.add_edge("synthesize", END)
```

更推荐的做法是用 `Send` API，更加灵活：

```python
from langgraph.constants import Send

def fan_out(state: State) -> list[Send]:
    """动态决定并发哪些 Node"""
    tasks = []
    if needs_kb_search(state):
        tasks.append(Send("search_kb", state))
    if needs_db_query(state):
        tasks.append(Send("query_db", state))
    return tasks

graph.add_conditional_edges("router", fan_out)
```

---

## 子图（Subgraph）

当 Agent 变复杂，一张大图会变得难以维护。子图让你把独立的逻辑封装成可复用的模块。

```python
# ============ RAG 子图 ============
class RAGState(TypedDict):
    query: str
    context: str
    answer: str

def retrieve(state: RAGState) -> dict:
    context = do_retrieval(state["query"])
    return {"context": context}

def generate(state: RAGState) -> dict:
    answer = do_generation(state["query"], state["context"])
    return {"answer": answer}

rag_graph = StateGraph(RAGState)
rag_graph.add_node("retrieve", retrieve)
rag_graph.add_node("generate", generate)
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "generate")
rag_subgraph = rag_graph.compile()


# ============ 主图 ============
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    rag_answer: str

def call_rag(state: MainState) -> dict:
    """在主图里调用 RAG 子图"""
    query = state["messages"][-1].content
    result = rag_subgraph.invoke({"query": query})
    return {"rag_answer": result["answer"]}

main_graph = StateGraph(MainState)
main_graph.add_node("rag", call_rag)
# ...
```

子图的好处：

- **复用**：同一个 RAG 子图可以在多个 Agent 里使用
- **隔离**：子图有自己的 State 类型，不会污染主图的 State
- **独立测试**：子图可以单独 invoke 验证

---

## Human-in-the-loop（人工审批）

生产 Agent 在关键节点需要人工审核后才能继续——比如执行危险操作、发送邮件、修改数据。

### 方式一：Interrupt（中断）

LangGraph 提供 `interrupt` 机制，在指定 Node 前暂停，等人工审批后继续。

```python
from langgraph.checkpoint.memory import MemorySaver

# 编译时加入 checkpointer（中断需要持久化状态）
checkpointer = MemorySaver()  # 内存版，生产用 PostgresSaver
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_action"]  # 在这个 Node 前暂停
)

# 运行到 execute_action 前会自动暂停
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(
    {"messages": [HumanMessage(content="删除所有测试数据")]},
    config=config
)
# result 返回暂停时的 State

# --- 人工审核 ---
# 人看到 Agent 要执行 "删除所有测试数据"，决定是否批准

# 批准：继续执行（传 None 表示不修改 State）
result = app.invoke(None, config=config)

# 拒绝：修改 State 后继续（比如把动作改成更安全的）
app.update_state(config, {
    "messages": [HumanMessage(content="仅删除 staging 环境的测试数据")]
})
result = app.invoke(None, config=config)
```

### 方式二：在 Node 内部请求人工输入

```python
from langgraph.types import interrupt

def execute_action(state: State) -> dict:
    action = state["planned_action"]

    # 需要人工确认
    human_response = interrupt(
        f"Agent 计划执行：{action}\n请回复 approve 或 reject"
    )

    if human_response == "approve":
        result = do_execute(action)
        return {"messages": [AIMessage(content=f"已执行：{result}")]}
    else:
        return {"messages": [AIMessage(content="操作已被人工取消")]}
```

---

## Checkpointing（状态持久化）

Checkpointer 在每个 Node 执行后保存完整的 State 快照。它解决两个问题：

1. **Human-in-the-loop**：中断后能从暂停点继续
2. **容错恢复**：Agent 崩溃后能从最近的 checkpoint 恢复

```python
# 内存版（开发用）
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# PostgreSQL 版（生产用）
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# SQLite 版（轻量生产用）
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("sqlite:///agent_state.db")

app = graph.compile(checkpointer=checkpointer)
```

使用 thread_id 区分不同会话的状态：

```python
# 每个用户/会话有独立的 thread_id
config = {"configurable": {"thread_id": "session-abc-123"}}

# 第一次运行
result = app.invoke({"messages": [HumanMessage(content="hello")]}, config=config)

# 同一个 thread_id 再次运行，会从上次的状态继续
result = app.invoke({"messages": [HumanMessage(content="接着上次的话题")]}, config=config)
```

### 查看历史状态

```python
# 获取某个 thread 的所有 checkpoint
history = list(app.get_state_history(config))
for state_snapshot in history:
    print(f"Step: {state_snapshot.metadata['step']}")
    print(f"Node: {state_snapshot.metadata.get('source')}")
    print(f"Messages count: {len(state_snapshot.values['messages'])}")
```

---

## Time Travel（时间旅行）

Checkpointer 记录了每一步的快照，可以回到过去某个时间点重新执行：

```python
# 获取历史状态
history = list(app.get_state_history(config))

# 回到第 3 步的状态
target_state = history[3]
target_config = target_state.config

# 从第 3 步重新开始执行（可以修改输入）
app.update_state(target_config, {
    "messages": [HumanMessage(content="换一种方式回答")]
})
result = app.invoke(None, target_config)
```

Time Travel 在调试时极其有用：找到 Agent 走错路的那个 Node，回到那个点修改输入，重新运行。

---

## 动态 Node 数量（Map-Reduce 模式）

有些场景需要动态创建 N 个并发 Node，比如"对 5 个文档分别做摘要，然后汇总"：

```python
from langgraph.constants import Send

def fan_out_docs(state: State) -> list[Send]:
    """为每个文档动态创建一个 summarize Node"""
    sends = []
    for doc in state["documents"]:
        sends.append(Send("summarize", {"document": doc}))
    return sends

def summarize(state: dict) -> dict:
    summary = do_summarize(state["document"])
    return {"summaries": [summary]}

def reduce(state: State) -> dict:
    combined = "\n".join(state["summaries"])
    return {"final_summary": combined}

graph.add_conditional_edges("load_docs", fan_out_docs)
graph.add_edge("summarize", "reduce")
```

---

## Streaming 进阶

LangGraph 的流式输出可以深入到 LLM 的 token 级别：

```python
# Node 级别的流式输出
for event in app.stream(inputs, stream_mode="updates"):
    for node, update in event.items():
        print(f"[{node}] completed")

# Token 级别的流式输出（看到 LLM 生成的每个 token）
async for event in app.astream_events(inputs, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            print(chunk.content, end="", flush=True)
    elif event["event"] == "on_tool_start":
        print(f"\n[calling tool: {event['name']}]")
```

---

## 生产级图结构示例

一个完整的 RAG Agent 图：

```python
class RAGAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str          # simple | complex | out_of_scope
    retrieval_results: list
    answer_quality: str      # good | needs_retry | failed

graph = StateGraph(RAGAgentState)

# Nodes
graph.add_node("classifier", classify_query)
graph.add_node("retriever", retrieve_documents)
graph.add_node("generator", generate_answer)
graph.add_node("quality_check", check_answer_quality)
graph.add_node("direct_answer", answer_without_retrieval)
graph.add_node("escalate", escalate_to_human)

# Entry
graph.set_entry_point("classifier")

# Classification routing
graph.add_conditional_edges("classifier", lambda s: s["query_type"], {
    "simple": "direct_answer",
    "complex": "retriever",
    "out_of_scope": "escalate"
})

# RAG flow
graph.add_edge("retriever", "generator")
graph.add_edge("generator", "quality_check")

# Quality gate
graph.add_conditional_edges("quality_check", lambda s: s["answer_quality"], {
    "good": END,
    "needs_retry": "retriever",   # 重新检索
    "failed": "escalate"          # 人工处理
})

graph.add_edge("direct_answer", END)
graph.add_edge("escalate", END)
```

图结构：

```text
                  ┌──────────────┐
                  │  classifier  │
                  └──────┬───────┘
               ┌─────────┼──────────┐
               ▼         ▼          ▼
        ┌────────┐ ┌──────────┐ ┌──────────┐
        │ direct │ │ retriever│ │ escalate │
        │ answer │ └────┬─────┘ └──────────┘
        └────┬───┘      ▼
             │    ┌──────────┐
             │    │generator │
             │    └────┬─────┘
             │         ▼
             │  ┌─────────────┐
             │  │quality_check│
             │  └──┬───┬───┬──┘
             │     │   │   │
             │  good retry failed
             │     │   │    │
             ▼     ▼   │    ▼
           ┌─────┐     │  ┌──────────┐
           │ END │ ◄───┘  │ escalate │
           └─────┘        └──────────┘
```

---

## 工程现场

场景：Agent 在 Human-in-the-loop 节点暂停后，长时间没有收到人工响应，State 丢失了。

原因：使用了 `MemorySaver`（内存版 checkpointer），进程重启后内存清空。

修复：生产环境必须使用持久化 checkpointer（Postgres / SQLite）。同时设置超时机制：

```python
import asyncio

async def wait_for_human_with_timeout(config, timeout_seconds=3600):
    """等待人工响应，超时后自动拒绝"""
    try:
        human_input = await asyncio.wait_for(
            get_human_input(config),
            timeout=timeout_seconds
        )
        return human_input
    except asyncio.TimeoutError:
        app.update_state(config, {
            "messages": [HumanMessage(content="人工审批超时，自动取消操作")]
        })
        return app.invoke(None, config)
```

---

## 小结

进阶能力总结：

| 能力 | 解决什么 | 核心 API |
|------|---------|---------|
| 并发节点 | 独立任务并行，降低延迟 | `Send` |
| 子图 | 复杂逻辑复用和隔离 | 子 `StateGraph` + `compile()` |
| Human-in-the-loop | 关键节点人工审批 | `interrupt_before` + `interrupt` |
| Checkpointing | 状态持久化、断点续跑 | `MemorySaver` / `PostgresSaver` |
| Time Travel | 调试时回到任意历史节点 | `get_state_history()` |
| Map-Reduce | 动态并发 N 个任务 | `Send` 列表 |

这些能力组合起来，可以构建从简单问答到复杂审批流程的任何 Agent。

下一篇讲 Multi-Agent：多个 Agent 如何协作。
