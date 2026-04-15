# Agent Loop：让 Agent 持续推理和行动

单次 Tool Calling 是"LLM 调用一次工具"。Agent Loop 是"LLM 反复调用工具，直到任务完成"。

这个区别看起来只是"加了个 while 循环"，但它带来了质的变化：Agent 可以把复杂任务拆解成多步，每步的结果影响下一步的决策。这是 Agent 能处理真实复杂任务的根本原因。

---

## ReAct：Agent Loop 的理论基础

ReAct（2022）是 Agent Loop 最重要的论文。核心思想是：

```text
Reason（推理）→ Act（行动）→ Observe（观察）→ Reason → ...
```

每一轮循环：

1. LLM 推理：当前情况是什么，下一步应该做什么
2. LLM 行动：生成 tool call，指定要调用的工具和参数
3. 执行工具：你的代码运行工具，得到结果
4. 观察结果：把结果追加到 messages，LLM 看到后继续推理

在此之前，LLM 只能用 Chain-of-Thought（纯推理，没有行动）。ReAct 让推理和外部动作交织在一起，大幅扩展了 LLM 能解决的任务范围。

---

## 最简 Agent Loop 实现

先用最少的代码手写一个能跑的 Agent Loop，不依赖任何框架：

```python
from openai import OpenAI
import json

client = OpenAI()

def run_agent(user_input: str, tools: list, tool_executors: dict, max_steps: int = 10):
    messages = [
        {"role": "system", "content": "你是一个助手，使用工具来回答用户问题。"},
        {"role": "user", "content": user_input}
    ]

    for step in range(max_steps):
        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        # 没有 tool call，说明 LLM 认为任务完成
        if not message.tool_calls:
            return message.content

        # 执行所有工具调用
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # 执行工具
            if func_name in tool_executors:
                result = tool_executors[func_name](**args)
            else:
                result = f"错误：工具 {func_name} 不存在"

            # 把结果追加到 messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    # 超过最大步数，强制返回
    return "已达到最大步数限制，任务未完成。"

# 使用
tool_executors = {
    "search_knowledge_base": search_knowledge_base,
    "get_weather": get_weather,
}

result = run_agent(
    user_input="RAG 系统中 rerank 的作用是什么？",
    tools=tools,
    tool_executors=tool_executors
)
```

这 40 行代码就是一个完整的 Agent。所有框架的 Agent 核心逻辑和这个本质上一样，只是加了更多的错误处理、可观测性、和配置选项。

---

## 核心工程问题

### 1. 终止条件

Agent Loop 必须有明确的终止条件，否则会无限运行。

```python
# 终止条件 1：LLM 不再调用工具（认为任务完成）
if not message.tool_calls:
    return message.content

# 终止条件 2：达到最大步数
if step >= max_steps:
    return "达到最大步数"

# 终止条件 3：检测到明确的完成信号（可选）
if "TASK_COMPLETE" in message.content:
    return extract_final_answer(message.content)
```

`max_steps` 的合理范围：简单任务 5-10 步，复杂任务 20-30 步。超过这个范围通常意味着 Agent 陷入了循环，继续跑下去没有意义。

### 2. 检测 Agent 陷入循环

Agent 有时会反复调用同一个工具，参数几乎相同，但结果没有推进任务。

```python
from collections import Counter

def detect_loop(messages: list, window: int = 6) -> bool:
    """检测最近 window 条消息里是否有重复的工具调用"""
    recent_tool_calls = []
    for msg in messages[-window:]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                recent_tool_calls.append(
                    f"{tc.function.name}:{tc.function.arguments}"
                )

    # 如果同一个调用出现超过 2 次，认为陷入循环
    counts = Counter(recent_tool_calls)
    return any(count > 2 for count in counts.values())

# 在 Loop 里使用
if detect_loop(messages):
    # 给 LLM 一个提示，让它换个方向
    messages.append({
        "role": "user",
        "content": "你似乎陷入了循环。请换一种方式来解决这个问题，或者告诉用户当前遇到的困难。"
    })
```

### 3. Context Window 管理

Agent Loop 运行多步后，messages 会越来越长，可能超出 context window。

```python
def trim_messages(messages: list, max_tokens: int = 100000) -> list:
    """保留 system prompt + 最近的 N 条消息"""
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    # 简单策略：保留最近的消息
    # 注意：要保持 tool call 和 tool result 的配对完整
    trimmed = other_messages[-20:]  # 保留最近 20 条

    # 确保第一条不是孤立的 tool result
    while trimmed and trimmed[0].get("role") == "tool":
        trimmed = trimmed[1:]

    return system_messages + trimmed
```

更精细的做法是用 tiktoken 精确计算 token 数，但对于大多数场景，"保留最近 N 条"已经够用。

### 4. 工具执行的异步化

生产环境里工具调用往往有 IO（HTTP 请求、数据库查询），必须异步：

```python
import asyncio

async def run_agent_async(user_input: str, tools: list, tool_executors: dict, max_steps: int = 10):
    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": user_input}
    ]

    for step in range(max_steps):
        response = await async_llm_call(messages, tools)
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        # 并发执行所有工具
        tasks = [execute_tool_async(tc, tool_executors) for tc in message.tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tool_call, result in zip(message.tool_calls, results):
            if isinstance(result, Exception):
                content = f"工具执行失败：{str(result)}"
            else:
                content = str(result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content
            })

    return "已达到最大步数限制。"
```

---

## 在 System Prompt 里引导 Agent 行为

System Prompt 对 Agent 行为的影响比大多数人意识到的更大。

```python
AGENT_SYSTEM_PROMPT = """你是一个知识库问答助手。

工作原则：
1. 在回答任何关于产品、技术的问题前，必须先调用 search_knowledge_base 工具
2. 如果搜索结果不足以回答问题，可以调用 search_knowledge_base 再次搜索，换不同关键词
3. 最多搜索 3 次，之后基于已有信息给出最佳答案
4. 如果确实不知道，直接说"我没有找到相关信息"，不要猜测

输出格式：
- 先简要说明你找到的信息来源
- 然后给出具体答案
- 如果有不确定的地方，明确指出
"""
```

好的 System Prompt 能减少 Agent 的"随机行为"，让行为更可预测、更易调试。

---

## Streaming 输出中的 Agent Loop

用户等待 Agent 多步执行时，需要流式输出中间状态，而不是等所有步骤完成后再一次性返回。

```python
async def run_agent_with_stream(user_input: str):
    messages = [{"role": "user", "content": user_input}]

    for step in range(max_steps):
        # 通知前端当前步骤
        yield f"data: {json.dumps({'type': 'step', 'step': step + 1})}\n\n"

        response = await async_llm_call(messages, tools)
        message = response.choices[0].message

        if not message.tool_calls:
            # 流式输出最终答案
            async for chunk in stream_final_answer(messages):
                yield f"data: {json.dumps({'type': 'answer', 'chunk': chunk})}\n\n"
            return

        # 通知前端正在调用的工具
        for tc in message.tool_calls:
            yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc.function.name})}\n\n"

        # 执行工具...
```

Streaming 在 Agent 里比普通 LLM 更重要，因为 Agent 可能运行 10-30 秒，没有中间反馈用户会以为系统卡了。

---

## 工程现场

场景：Agent 在步骤 3 拿到了正确的工具结果，但步骤 4 又去重复搜索同样的内容，浪费了 2-3 步。

根本原因：LLM 没有看清楚之前的工具结果，或者 messages 太长导致关键结果被忽略。

修复方向：

1. 工具返回值加 summary 字段，让关键信息更显眼。
2. 在每步开始前，给 LLM 一个"当前已知信息摘要"，减少 LLM 重复搜索的冲动。
3. 检测到重复 tool call 时，注入一条 user 消息提醒 LLM。

---

## 小结

手写 Agent Loop 的目的不是为了在生产里用这段代码，而是为了理解：

- Agent 的本质是 messages 列表的持续增长——每步推理和行动都在追加消息
- 终止条件、循环检测、context 管理是工程上必须处理的三个问题
- 框架（LangGraph、LangChain）解决的正是这些问题，理解了原理才能用好框架

下一篇讲 Memory：在 Agent Loop 里，如何管理短期记忆、长期记忆和任务执行状态。
