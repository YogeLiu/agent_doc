# Tool Calling：Agent 行动能力的基础

很多人第一次看到 Tool Calling 时，觉得这只是"让 LLM 调用函数"。这个理解不完整。

Tool Calling 真正解决的问题是：**如何让 LLM 的输出从"自然语言"变成"可执行指令"**。

没有 Tool Calling，LLM 只能生成文字。有了 Tool Calling，LLM 能触发真实的系统动作。这是 Agent 和普通 LLM 应用之间的根本区别。

---

## 工作原理

Tool Calling 的核心机制很简单：

```text
1. 你定义工具（函数名 + 参数 schema）
2. 把工具描述连同用户请求一起发给 LLM
3. LLM 决定是否调用工具，以及用什么参数
4. LLM 返回结构化的 tool call（JSON），不是自然语言
5. 你的代码执行工具，把结果追加到 messages
6. LLM 看到结果后继续推理
```

整个过程 LLM 只负责"决策"，不负责"执行"。执行永远在你的代码里。

---

## 定义一个工具

以 OpenAI 格式为例（大多数 LLM API 兼容这个格式）：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中搜索相关信息。当用户问到具体知识点时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

description 是关键。LLM 靠 description 来判断"该不该用这个工具、什么时候用"。description 写得模糊，LLM 就会乱用或不用。

---

## 一次完整的 Tool Call 交互

```python
from openai import OpenAI

client = OpenAI()

messages = [
    {"role": "user", "content": "RAG 系统中 rerank 是什么？"}
]

# 第一次调用：LLM 决定是否用工具
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

# 检查 LLM 是否要调用工具
if message.tool_calls:
    tool_call = message.tool_calls[0]

    # 解析参数
    import json
    args = json.loads(tool_call.function.arguments)
    # args = {"query": "RAG rerank"}

    # 执行工具（你的代码）
    result = search_knowledge_base(args["query"])

    # 把工具结果追加到 messages
    messages.append(message)  # LLM 的 tool call 请求
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })

    # 第二次调用：LLM 基于工具结果生成最终回答
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )

    print(final_response.choices[0].message.content)
```

注意 messages 的结构。每次 tool call 都要追加两条消息：

1. LLM 发出的 tool call 请求（role: messages.append(response.message)）
2. 工具执行的结果（role: tool）

原因：

1. 不符合 API 协议
2. 不符合模型训练分布， 效果会变差

---

## 并行工具调用

LLM 可以在一次回复里请求多个工具调用，用于处理"可以并行做的事情"。

```python
# LLM 可能返回多个 tool_calls
if message.tool_calls:
    import asyncio

    async def execute_tool(tool_call):
        args = json.loads(tool_call.function.arguments)
        if tool_call.function.name == "search_knowledge_base":
            return await async_search(args["query"])
        elif tool_call.function.name == "get_weather":
            return await async_get_weather(args["city"])

    # 并发执行所有工具
    results = await asyncio.gather(
        *[execute_tool(tc) for tc in message.tool_calls]
    )

    # 把所有结果追加到 messages
    messages.append(message)
    for tool_call, result in zip(message.tool_calls, results):
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })
```

并行工具调用是 Agent 性能优化的关键。如果 LLM 需要查三个独立数据源，串行执行是 3 倍延迟，并发执行接近 1 倍。

---

## 工具失败处理

工具出错时，不应该让整个 Agent 崩溃，而是把错误信息作为观察结果返回给 LLM，让 LLM 决定下一步。

```python
async def execute_tool_safe(tool_call):
    try:
        args = json.loads(tool_call.function.arguments)
        result = await execute_tool(tool_call)
        return {
            "tool_call_id": tool_call.id,
            "content": str(result),
            "status": "success"
        }
    except json.JSONDecodeError:
        return {
            "tool_call_id": tool_call.id,
            "content": "参数解析失败，请检查参数格式",
            "status": "error"
        }
    except Exception as e:
        return {
            "tool_call_id": tool_call.id,
            "content": f"工具执行失败：{str(e)}",
            "status": "error"
        }
```

LLM 看到错误信息后通常能做出合理的下一步决策：换个参数重试、换个工具、或者告诉用户遇到了问题。

---

## Tool Schema 设计原则

工具设计的好坏直接影响 Agent 的可靠性。

**原则一：description 要说清楚"什么时候用"**

```python
# 差
"description": "搜索信息"

# 好
"description": "在内部知识库中搜索技术文档和产品信息。适用于回答关于产品功能、技术规格、操作指南的问题。不适合搜索实时新闻或股价等时效性信息。"
```

**原则二：参数类型要明确，枚举值要列出来**

```python
# 差
"format": {"type": "string", "description": "输出格式"}

# 好
"format": {
    "type": "string",
    "enum": ["json", "markdown", "plain_text"],
    "description": "输出格式：json 适合结构化处理，markdown 适合展示，plain_text 适合简单文本"
}
```

**原则三：工具粒度要合适——一个工具只做一件事**

```python
# 差：一个工具做太多事
"name": "manage_file",  # read / write / delete 全在里面

# 好：分开成独立工具
"name": "read_file"
"name": "write_file"
"name": "delete_file"
```

粒度太粗，LLM 容易搞混参数。粒度太细，工具数量爆炸，LLM 反而不知道选哪个。通常一个 Agent 工具集保持在 5-15 个之间是比较合理的范围。

**原则四：返回值要结构化，加上状态标记**

```python
# 差
return "找到了一些结果"

# 好
return {
    "status": "success",
    "results": [...],
    "total": 5,
    "source": "knowledge_base"
}
```

LLM 需要明确的信号来判断工具是否成功执行。模糊的返回值会让 LLM 判断失误。

---

## tool_choice 控制

`tool_choice` 参数控制 LLM 如何使用工具：

```python
# auto（默认）：LLM 自行决定用不用工具
tool_choice="auto"

# required：必须使用工具（不允许直接回答）
tool_choice="required"

# none：禁止使用工具（只生成文字）
tool_choice="none"

# 强制使用特定工具
tool_choice={
    "type": "function",
    "function": {"name": "search_knowledge_base"}
}
```

在 Agent Loop 的某些阶段，强制调用特定工具能让行为更可预测。比如"任务规划"阶段，可以强制 LLM 调用 `plan_task` 工具，保证输出结构化的任务计划。

---

## 工程现场

场景：Tool Calling 在生产中偶发性地"不调用工具"，LLM 直接生成文字回答，绕过了工具。

常见原因：

1. **description 写得太模糊**，LLM 不确定该用，直接猜一个答案。
2. **工具太多**（超过 20 个），LLM 在选择时"放弃"，直接回答。
3. **messages history 太长**，关键的 system prompt 被稀释，LLM 忘记了有工具可用。

修复方向：

1. 在 system prompt 里明确说明"在回答 XX 类问题时必须调用 search 工具"。
2. 工具数量超过 10 个时，考虑按场景分组，动态加载当前需要的工具子集。
3. 设 `tool_choice="required"` 强制工具调用，适合明确需要检索才能回答的场景。

---

## 小结

Tool Calling 是 Agent 的行动基础：

- LLM 负责"决策"：用哪个工具、传什么参数
- 你的代码负责"执行"：真正运行工具、返回结果
- messages 是桥梁：工具结果通过 messages 传回给 LLM

下一步是把单次 Tool Call 组合成一个循环，让 Agent 能多步推理和行动——这就是 Agent Loop。
