# Multi-Agent：多个 Agent 协作

单个 Agent 处理所有事情，到一定复杂度之后会出问题：工具太多选不对、System Prompt 过长导致行为不稳定、一个功能出 bug 影响全部。

Multi-Agent 的思路是：**把一个大 Agent 拆成多个小 Agent，每个专注一件事，通过协作完成复杂任务。**

---

## 什么时候需要 Multi-Agent

不是所有场景都需要 Multi-Agent。判断标准：

```text
需要 Multi-Agent：
- 工具超过 15 个，LLM 经常选错工具
- System Prompt 超过 2000 token，行为开始不稳定
- 任务包含多个独立领域（如检索 + 代码生成 + 数据分析）
- 需要不同模型处理不同任务（如 GPT-4o 做规划，GPT-4o-mini 做执行）

不需要 Multi-Agent：
- 工具少于 10 个，LLM 选择准确
- 任务单一领域
- 延迟敏感（Multi-Agent 会增加通信开销）
```

---

## 两种核心模式

### Supervisor 模式

一个中心 Agent（Supervisor）负责调度，多个 Worker Agent 负责执行。

```text
         ┌──────────────┐
         │  Supervisor  │ ← 决定下一步给谁
         └──────┬───────┘
         ┌──────┼──────┐
         ▼      ▼      ▼
    ┌────────┐ ┌────┐ ┌───────┐
    │Researcher│ │Coder│ │Reviewer│
    └────────┘ └────┘ └───────┘
```

Supervisor 是一个 LLM Agent，它看到用户请求和当前进展后，决定把任务交给哪个 Worker。

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str


# Supervisor：决定下一步给谁
supervisor_llm = ChatOpenAI(model="gpt-4o")

def supervisor(state: MultiAgentState) -> dict:
    system = """你是一个任务调度器。根据用户的请求和当前进展，决定下一步交给哪个 Agent：

- researcher：需要查找信息、检索文档时
- coder：需要编写或修改代码时
- reviewer：需要审核代码或结果时
- FINISH：任务已完成

只回复 Agent 名称。"""

    response = supervisor_llm.invoke(
        [{"role": "system", "content": system}] + state["messages"]
    )
    return {"next_agent": response.content.strip().lower()}


# Worker Agent：各自有独立的工具和 System Prompt
def create_worker(name: str, system_prompt: str, tools: list):
    worker_llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

    def worker(state: MultiAgentState) -> dict:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = worker_llm.invoke(messages)

        # Worker 的回复带上身份标记
        response.name = name
        return {"messages": [response]}

    return worker


researcher = create_worker(
    name="researcher",
    system_prompt="你是一个研究助手，负责检索和整理信息。",
    tools=[search_knowledge_base, web_search]
)

coder = create_worker(
    name="coder",
    system_prompt="你是一个代码工程师，负责编写和修改代码。",
    tools=[read_file, write_file, run_code]
)

reviewer = create_worker(
    name="reviewer",
    system_prompt="你是一个代码审查员，负责检查代码质量和正确性。",
    tools=[read_file, run_tests]
)


# 构建图
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("reviewer", reviewer)

# Supervisor 根据决策路由到不同 Worker
def route_to_worker(state: MultiAgentState) -> str:
    next_agent = state["next_agent"]
    if next_agent == "finish":
        return END
    return next_agent

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_to_worker)

# 每个 Worker 完成后回到 Supervisor
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
graph.add_edge("reviewer", "supervisor")

app = graph.compile()
```

Supervisor 模式的优缺点：

```text
优点：
- 中心化控制，容易理解任务流向
- Supervisor 可以根据全局状态做最优决策
- 适合任务动态分配的场景

缺点：
- Supervisor 本身是性能瓶颈——每步都要过 Supervisor
- Supervisor 选错 Worker 时，影响整条链
- 不适合 Worker 之间需要频繁通信的场景
```

---

### Swarm 模式（Handoff）

没有中心调度器，Agent 之间直接移交控制权。每个 Agent 自己决定什么时候把任务交给下一个。

```text
Agent A ──handoff──▶ Agent B ──handoff──▶ Agent C
```

OpenAI 的 Swarm 框架推广了这个模式。核心思想：每个 Agent 有一个特殊工具 `transfer_to_X`，调用后控制权转移。

```python
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# Agent 之间的移交通过 Command 实现
def transfer_to_coder():
    """当需要编写代码时，调用此工具将任务转给代码工程师"""
    return Command(goto="coder")

def transfer_to_reviewer():
    """当代码写完需要审查时，调用此工具将任务转给审查员"""
    return Command(goto="reviewer")

def transfer_to_researcher():
    """当需要查找信息时，调用此工具将任务转给研究助手"""
    return Command(goto="researcher")

# 每个 Agent 有自己的工具 + transfer 工具
researcher_agent = create_react_agent(
    model="gpt-4o",
    tools=[search_knowledge_base, web_search, transfer_to_coder],
    prompt="你是一个研究助手。找到足够信息后，转给 coder 写代码。"
)

coder_agent = create_react_agent(
    model="gpt-4o",
    tools=[read_file, write_file, run_code, transfer_to_reviewer],
    prompt="你是一个代码工程师。代码写完后，转给 reviewer 审查。"
)

reviewer_agent = create_react_agent(
    model="gpt-4o",
    tools=[read_file, run_tests, transfer_to_researcher, transfer_to_coder],
    prompt="你是一个审查员。发现问题转给 coder 修复，需要更多信息转给 researcher。"
)

# 构建图
graph = StateGraph(AgentState)
graph.add_node("researcher", researcher_agent)
graph.add_node("coder", coder_agent)
graph.add_node("reviewer", reviewer_agent)

graph.set_entry_point("researcher")
# Edge 由 Command(goto=...) 动态决定
```

Swarm 模式的优缺点：

```text
优点：
- 去中心化，没有 Supervisor 瓶颈
- Agent 之间直接通信，延迟更低
- 每个 Agent 更自主，职责更清晰

缺点：
- 缺乏全局视角，可能出现 Agent 之间互相推诿
- 调试困难——要跟踪控制权在多个 Agent 之间的跳转
- 需要精心设计 transfer 的时机和条件
```

---

## 选择 Supervisor 还是 Swarm

| 维度 | Supervisor | Swarm |
|------|-----------|-------|
| 控制流 | 中心化，Supervisor 全局决策 | 去中心化，Agent 自主移交 |
| 适合场景 | 任务动态分配，Worker 相对独立 | 流程固定，Agent 顺序明确 |
| 延迟 | 每步多一次 Supervisor 调用 | Agent 直接移交，延迟更低 |
| 调试 | 查 Supervisor 决策日志 | 跟踪 handoff 链路 |
| 典型用例 | 智能客服分派、复杂分析任务 | 文档处理流水线、代码审查流程 |

实际项目中可以组合使用：Supervisor 做顶层调度，Swarm 在局部子流程中使用。

---

## State 共享与隔离

Multi-Agent 最大的工程问题：多个 Agent 的 State 怎么管理。

### 共享 State

所有 Agent 读写同一个 State，通过 messages 列表通信：

```python
# 所有 Agent 共享 messages
class SharedState(TypedDict):
    messages: Annotated[list, add_messages]

# 问题：Agent A 的中间推理污染了 Agent B 的 messages
# Agent B 看到大量与自己无关的 messages，反而干扰判断
```

### 隔离 State + 摘要传递

每个 Agent 有自己的局部 State，只通过摘要互相传递信息：

```python
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    research_summary: str      # Researcher 的产出摘要
    code_output: str           # Coder 的产出
    review_result: str         # Reviewer 的评审结果

def researcher(state: MainState) -> dict:
    # Researcher 只看 messages 里和自己相关的部分
    research_result = do_research(state["messages"])
    # 输出摘要，而不是完整的中间过程
    return {"research_summary": summarize(research_result)}

def coder(state: MainState) -> dict:
    # Coder 看 research_summary，而不是 Researcher 的所有 messages
    code = write_code(state["research_summary"])
    return {"code_output": code}
```

**推荐做法**：messages 共享（用于全局上下文），但每个 Agent 有独立的结构化字段存放产出。避免一个 Agent 的大量中间消息淹没另一个 Agent 的输入。

---

## 生产模式：带兜底的 Multi-Agent

生产环境需要处理 Agent 之间的异常：

```python
def supervisor_with_fallback(state: MultiAgentState) -> dict:
    max_delegations = 5  # 最多分配 5 次
    delegation_count = state.get("delegation_count", 0)

    if delegation_count >= max_delegations:
        # 兜底：直接回答，不再分配
        return {
            "next_agent": "finish",
            "messages": [AIMessage(content="任务过于复杂，建议拆分后重试。")]
        }

    decision = get_supervisor_decision(state)
    return {
        "next_agent": decision,
        "delegation_count": delegation_count + 1
    }

# 每个 Worker 也需要超时和错误处理
def worker_with_timeout(worker_fn, timeout_seconds=60):
    async def wrapped(state):
        try:
            return await asyncio.wait_for(
                worker_fn(state),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return {
                "messages": [AIMessage(content="[Worker 超时，跳过此步骤]")]
            }
    return wrapped
```

---

## 工程现场

场景：Supervisor Agent 在"researcher"和"coder"之间反复跳转，每次 Supervisor 说"需要更多信息"→ researcher，然后又说"可以开始写代码了"→ coder，循环 10 多次。

原因：Supervisor 的 System Prompt 没有看到全局进展，每次只看最后一条消息就做决策。

修复：

1. 在 State 里加 `completed_steps` 列表，记录每个 Worker 已完成的工作。
2. Supervisor 的 prompt 里注入已完成步骤的摘要。
3. 加循环检测：如果同一个 Worker 被连续分配超过 2 次，强制切到下一个 Worker。

```python
def supervisor(state: MultiAgentState) -> dict:
    completed = state.get("completed_steps", [])
    completed_summary = "\n".join([f"- {s}" for s in completed])

    system = f"""你是任务调度器。

已完成的步骤：
{completed_summary}

不要重复分配已完成的步骤。根据当前进展决定下一步。"""

    # ...
```

---

## 小结

| 模式 | 适用场景 | 关键特征 |
|------|---------|---------|
| Supervisor | 动态任务分配 | 中心调度，全局视角 |
| Swarm | 固定流程流转 | 去中心化，Agent 自主移交 |
| 混合 | 复杂系统 | 顶层 Supervisor + 局部 Swarm |

设计 Multi-Agent 时最重要的决策不是选哪个模式，而是**如何划分 Agent 的职责边界**和**如何管理 Agent 之间的状态传递**。

下一篇讲 MCP：当工具数量增长后，如何标准化工具的定义和调用。
