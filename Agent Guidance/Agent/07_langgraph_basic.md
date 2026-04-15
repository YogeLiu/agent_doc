# LangGraph 基础：用状态图构建 Agent

手写 Agent Loop 让你理解了原理，但生产级 Agent 需要更多能力：条件分支、并发执行、状态持久化、人工审批。自己全写一遍不是不行，但没有必要。

LangGraph 把 Agent 建模为**状态图（StateGraph）**，提供了这些能力的标准实现。

---

## 为什么是 LangGraph 而不是 LangChain

LangChain 的 AgentExecutor 是黑盒：你给它工具和 LLM，它内部跑一个 ReAct Loop，你很难控制中间流程。

LangGraph 是白盒：你显式定义每一步是什么、步骤之间怎么连接、状态怎么流转。工程上更可控、更易调试。

```text
LangChain AgentExecutor：
    LLM + Tools → 黑盒循环 → 结果

LangGraph：
    你定义 Node（做什么）
    你定义 Edge（走哪条路）
    你定义 State（带什么数据）
    框架负责运行这个图
```

LangGraph 不是 LangChain 的替代品，它是 LangChain 生态里专门解决 Agent 编排问题的组件。

---

## 核心概念：三件套

### State（状态）

State 是整个图的共享数据结构，所有 Node 读写同一个 State。

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # messages 是 Agent 的核心状态
    # add_messages 是 reducer：新消息追加到列表，不是覆盖
    messages: Annotated[list, add_messages]

    # 可以加任何自定义字段
    current_step: str
    retrieval_count: int
```

**关键概念：Reducer**

State 的每个字段可以指定一个 reducer 函数，决定"当 Node 返回新值时，如何与旧值合并"。

```python
# add_messages reducer：新消息追加到列表尾部
messages: Annotated[list, add_messages]

# 默认行为（无 reducer）：新值直接覆盖旧值
current_step: str  # Node 返回 {"current_step": "plan"} 就覆盖

# 自定义 reducer：比如计数器累加
from operator import add
total_tokens: Annotated[int, add]  # Node 返回 {"total_tokens": 150} 就累加
```

Reducer 解决了一个重要问题：多个 Node 同时写同一个字段时，如何合并。没有 reducer 只是覆盖，有 reducer 可以追加、累加、合并。

---

### Node（节点）

Node 是一个函数，输入 State，输出 State 的部分更新。

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(model="gpt-4o")

# Node 1：调用 LLM
def call_llm(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    # 返回的 dict 会被合并到 State 里
    # messages 字段有 add_messages reducer，所以是追加
    return {"messages": [response]}

# Node 2：执行工具（LangGraph 内置的 ToolNode）
tool_node = ToolNode(tools=[search_knowledge_base, get_weather])
```

Node 的关键约束：
- 输入是完整的 State
- 输出是 State 的部分更新（dict），不是完整 State
- Node 内部不应该有副作用（除了工具执行），保持纯函数风格

---

### Edge（边）

Edge 决定 Node 之间的转移关系。

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# 添加 Node
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)

# 普通 Edge：从 tools 固定走到 llm
graph.add_edge("tools", "llm")

# 条件 Edge：从 llm 根据条件走不同路
def should_use_tool(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"      # 有 tool call，去执行工具
    return END               # 没有 tool call，结束

graph.add_conditional_edges("llm", should_use_tool)

# 入口
graph.set_entry_point("llm")

# 编译
app = graph.compile()
```

---

## 第一个完整示例：ReAct Agent

用 LangGraph 实现 02_agent_loop.md 里手写的 ReAct Agent：

```python
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# 1. 定义 State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. 定义工具
tools = [search_knowledge_base, get_weather]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# 3. 定义 Node
def call_llm(state: AgentState) -> dict:
    system = SystemMessage(content="你是一个知识库问答助手。")
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


# 4. 定义路由
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# 5. 构建图
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")

app = graph.compile()


# 6. 运行
from langchain_core.messages import HumanMessage

result = app.invoke({
    "messages": [HumanMessage(content="RAG 系统中 rerank 的作用是什么？")]
})

print(result["messages"][-1].content)
```

画出来的图结构：

```text
       ┌──────┐
       │  llm │ ◄──────────┐
       └──┬───┘            │
          │                │
    ┌─────┴──────┐         │
    │            │         │
has_tool_call  no_tool_call│
    │            │         │
    ▼            ▼         │
 ┌──────┐     ┌─────┐     │
 │tools │────▶│     │     │
 └──────┘     │ END │     │
              └─────┘     │
```

这和手写的 Agent Loop 逻辑完全一样，但 LangGraph 额外提供了：状态持久化、流式输出、可视化、断点续跑。

---

## 自定义 State：加入 Working Memory

除了 messages，可以在 State 里加任何字段：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    # Working Memory 字段
    task_plan: list[str]              # 任务计划
    completed_steps: list[str]        # 已完成步骤
    current_step_index: int           # 当前步骤索引
    retrieval_results: dict           # 检索结果缓存

    # 元数据
    step_count: int
    total_tokens: Annotated[int, add]  # 用 add reducer 累加
```

Node 可以读写这些字段：

```python
def planner(state: AgentState) -> dict:
    """规划任务步骤"""
    plan = generate_plan(state["messages"])
    return {
        "task_plan": plan,
        "current_step_index": 0
    }

def executor(state: AgentState) -> dict:
    """执行当前步骤"""
    step = state["task_plan"][state["current_step_index"]]
    result = execute_step(step, state["messages"])
    return {
        "messages": [result],
        "completed_steps": state["completed_steps"] + [step],
        "current_step_index": state["current_step_index"] + 1,
        "step_count": state["current_step_index"] + 1
    }
```

---

## 条件路由：让 Agent 动态决策

条件路由是 LangGraph 最强大的特性之一。它让 Agent 根据当前状态动态选择下一步。

```python
def route_after_plan(state: AgentState) -> str:
    plan = state.get("task_plan", [])
    if not plan:
        return "direct_answer"    # 没有计划，直接回答
    elif len(plan) == 1:
        return "simple_executor"  # 只有一步，用简单执行器
    else:
        return "multi_step"       # 多步骤，用多步执行器

graph.add_conditional_edges("planner", route_after_plan, {
    "direct_answer": "llm",
    "simple_executor": "executor",
    "multi_step": "multi_step_executor"
})
```

可以看出，条件路由就是一个普通函数，输入 State，返回下一个 Node 的名字。LangGraph 支持把返回值映射到 Node 名称，让代码更清晰。

---

## 流式输出

LangGraph 原生支持流式输出，可以看到每个 Node 的执行过程：

```python
# stream 方法逐步返回每个 Node 的输出
for event in app.stream(
    {"messages": [HumanMessage(content="查一下 rerank 的原理")]},
    stream_mode="updates"  # 只返回 State 的增量更新
):
    for node_name, state_update in event.items():
        print(f"[{node_name}] →", state_update)

# 输出示例：
# [llm] → {"messages": [AIMessage(tool_calls=[...])]}
# [tools] → {"messages": [ToolMessage(content="...")]}
# [llm] → {"messages": [AIMessage(content="rerank 是...")]}
```

stream_mode 选项：

```python
# "values" — 每步返回完整 State（数据量大）
# "updates" — 每步只返回 State 的变化部分（推荐）
# "messages" — 只返回 messages 的变化（最轻量）
```

---

## 可视化

LangGraph 可以把图结构导出为图片，方便调试和文档化：

```python
from IPython.display import Image

# 输出 Mermaid 格式
print(app.get_graph().draw_mermaid())

# 输出 PNG 图片
Image(app.get_graph().draw_mermaid_png())
```

---

## 常见图模式

### 模式一：简单 ReAct Loop

```text
llm ←→ tools → END
```

最基础的 Agent，上面已经实现。

### 模式二：Plan-and-Execute

```text
planner → executor → checker → (回到 executor 或 END)
```

```python
graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("checker", checker)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "checker")

def check_result(state: AgentState) -> str:
    if state["current_step_index"] >= len(state["task_plan"]):
        return END               # 所有步骤完成
    return "executor"             # 继续下一步

graph.add_conditional_edges("checker", check_result)
```

### 模式三：Router

```text
classifier → (rag_agent | direct_answer | escalate_to_human)
```

```python
def classify_intent(state: AgentState) -> str:
    intent = classify(state["messages"][-1].content)
    if intent == "knowledge_question":
        return "rag_agent"
    elif intent == "simple_chat":
        return "direct_answer"
    else:
        return "escalate"

graph.set_entry_point("classifier")
graph.add_conditional_edges("classifier", classify_intent)
```

---

## 工程现场

场景：LangGraph 图编译成功，但运行时报 `State key 'messages' is not defined` 错误。

原因：Node 函数返回了一个不在 State TypedDict 里的 key，或者忘记给 messages 字段设置 reducer。

排查清单：

1. 确认 State TypedDict 里定义了所有 Node 会写入的 key
2. 确认 messages 字段用了 `Annotated[list, add_messages]`
3. 确认每个 Node 返回的是 dict（State 的部分更新），不是完整的 State 对象

---

## 小结

LangGraph 的核心模型：

```text
State — 图的共享数据，Node 之间的通信介质
Node  — 纯函数，输入 State 输出部分更新
Edge  — 连接 Node，条件 Edge 实现动态决策
Reducer — 定义 State 字段的合并策略
```

理解了这四个概念，就能构建任何 Agent 图。下一篇讲进阶用法：并发节点、子图、Human-in-the-loop、和状态持久化。
