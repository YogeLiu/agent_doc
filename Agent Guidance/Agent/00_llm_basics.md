# LLM 基础能力：驱动 Agent 的底层

Agent 的所有行为最终都落在 LLM 调用上。在进入 Tool Calling 和 Agent Loop 之前，需要先把这几件事搞清楚：如何让 LLM 稳定输出可执行的结果。

这不是训练模型，而是理解如何**用好一个已有的 LLM**。

---

## Prompt 的基本结构

一次 LLM 调用由三部分消息组成：

```text
system   — 角色定义、行为规则、输出格式要求（整个会话只设一次）
user     — 用户输入或任务描述
assistant — LLM 的回复（多轮对话时，历史回复也在这里）
```

```python
messages = [
    {"role": "system",    "content": "你是一个专业的技术文档助手。"},
    {"role": "user",      "content": "解释一下什么是 RAG"},
    {"role": "assistant", "content": "RAG 是检索增强生成..."},
    {"role": "user",      "content": "它的主要优势是什么？"}
]
```

多轮对话时，把历史消息完整传入。LLM 没有"记忆"，它只能看到当前传入的 messages。

---

## System Prompt

System Prompt 是 Agent 行为的控制中心。它在整个会话里持续生效，决定 LLM 的角色、风格和约束。

```python
SYSTEM_PROMPT = """你是一个代码审查助手，专注于 Python 后端代码。

职责：
- 找出代码中的安全漏洞和性能问题
- 给出具体的改进建议，附带代码示例
- 不评价代码风格，只关注功能正确性和安全性

输出格式：
1. 问题描述（一句话）
2. 风险等级（高/中/低）
3. 修复建议（代码示例）
"""
```

System Prompt 的核心原则：

- **说清楚"不做什么"**——比"做什么"更有效控制边界
- **格式要求放在 System Prompt 里**——比放在每次 user message 里更稳定
- **Agent 场景下，把工具使用规则写进 System Prompt**——"在回答技术问题前，必须先调用 search 工具"

---

## Role Prompting

Role Prompting 是通过给 LLM 一个明确角色来提升输出质量。

```python
# 不用角色
"请分析这段代码有什么问题"

# 用角色
"你是一位有 10 年经验的 Python 安全工程师，请分析这段代码有什么安全漏洞"
```

原因：角色设定激活了模型训练数据中对应角色的知识模式。"安全工程师"这个角色会让模型更多地关注安全相关的模式。

在 Agent 里的典型应用：

```python
system = """你是一个严格的任务规划师。
你的工作是把用户给出的目标分解为可执行的步骤。

要求：
- 每个步骤必须是具体可操作的，不能是模糊的方向性描述
- 步骤数量控制在 3-7 步之间
- 每个步骤说明：做什么、用什么工具、预期输出是什么
"""
```

---

## Few-shot Prompting

Few-shot 是给 LLM 提供几个示例，让它学会期望的输出格式和推理风格。

```python
messages = [
    {
        "role": "system",
        "content": "你是一个工单分类助手，把用户问题分类到对应部门。"
    },
    # 示例 1
    {"role": "user",      "content": "我的账单里有一笔重复扣费"},
    {"role": "assistant", "content": '{"department": "billing", "priority": "high"}'},
    # 示例 2
    {"role": "user",      "content": "我想修改我的用户名"},
    {"role": "assistant", "content": '{"department": "account", "priority": "low"}'},
    # 示例 3
    {"role": "user",      "content": "App 一直闪退，无法打开"},
    {"role": "assistant", "content": '{"department": "technical", "priority": "high"}'},
    # 真正的问题
    {"role": "user",      "content": "我的订单显示已发货但三天没有物流更新"}
]
```

Few-shot 的价值：

- 比在 system prompt 里写规则更有效——示例比描述更清晰
- 特别适合**格式复杂或分类边界模糊**的任务
- 示例质量比数量更重要，3 个好示例胜过 10 个模糊示例

在 Agent 里，few-shot 常用于让 LLM 学会特定格式的工具调用方式。

---

## Chain-of-Thought（CoT）

Chain-of-Thought 是让 LLM 在给出最终答案前，先写出推理过程。

```python
# 不用 CoT（直接要答案）
"这道数学题的答案是什么：一个商品打八折后售价 80 元，原价是多少？"
# LLM 可能直接回答，错误率较高

# 用 CoT（让 LLM 先推理）
"这道数学题的答案是什么：一个商品打八折后售价 80 元，原价是多少？请一步步推理。"
# LLM 会写出：
# 设原价为 x
# 打八折 = x * 0.8 = 80
# x = 80 / 0.8 = 100
# 原价是 100 元
```

两种实现方式：

**方式一：在 prompt 里加"一步步思考"**

```python
"请一步步分析这个问题，然后给出结论。"
"Let's think step by step."  # 英文场景更常用
```

**方式二：使用支持 Thinking 的模型（如 Claude 3.5 Sonnet）**

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8000,
    thinking={
        "type": "enabled",
        "budget_tokens": 5000  # 允许多少 token 用于内部推理
    },
    messages=[{"role": "user", "content": "分析这个系统的架构风险"}]
)
```

在 Agent 里，CoT 主要用于 Planner 步骤——让 LLM 在拆解任务前先分析清楚任务的边界和约束。

---

## Structured Output

对 Agent 来说，Structured Output 是最重要的能力之一。LLM 的输出必须是可解析的结构，而不是自由文字。

### 方式一：JSON Mode

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你只能输出合法的 JSON，不要有任何额外文字。"},
        {"role": "user",   "content": "提取以下文本的关键信息：张三，男，28岁，软件工程师"}
    ],
    response_format={"type": "json_object"}  # 强制 JSON 输出
)

import json
data = json.loads(response.choices[0].message.content)
# {"name": "张三", "gender": "男", "age": 28, "occupation": "软件工程师"}
```

JSON Mode 的问题是：它通常只保证"看起来像 JSON"，但不一定保证字段齐全、类型正确、枚举值合法。

### 方式二：Native Structured Outputs（模型原生结构化输出）

更进一步的做法，是直接把 schema 交给模型或平台，让它按 schema 产出结果。

这类能力可以统称为 **Native Structured Outputs**。重点是：**结构化约束是模型 API 原生支持的，不只是你在 prompt 里写规则**。

```python
from pydantic import BaseModel

class TaskPlan(BaseModel):
    goal: str
    steps: list[str]
    estimated_tools: list[str]
    priority: str  # "high" | "medium" | "low"

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个任务规划助手。"},
        {"role": "user",   "content": "帮我规划：写一篇关于 RAG 的技术博客"}
    ],
    response_format=TaskPlan  # 传入 Pydantic 模型
)

plan: TaskPlan = response.choices[0].message.parsed
# plan.goal = "写一篇关于 RAG 的技术博客"
# plan.steps = ["研究 RAG 技术", "整理资料", ...]
# plan.priority = "medium"
```

它的优势是：

- 比 JSON Mode 更稳定，不容易漏字段或输出脏文本
- 可以直接返回已解析的对象，减少手写解析代码
- 很适合 Agent 里的分类、计划生成、信息抽取等结构明确的任务

但要注意：**它约束的是输出结构，不是内容真实性**。字段格式再对，也可能一本正经地胡说八道。

### 方式三：schema-first libs（先定义 schema，再驱动 LLM）

很多工程里，开发者并不是直接手写 JSON Schema，而是先在代码里定义数据模型，再让库去驱动 LLM 输出。

这类做法可以叫 **schema-first**：先有 schema，再有 prompt、校验、解析和重试逻辑。

```python
from pydantic import BaseModel

class TaskPlan(BaseModel):
    goal: str
    steps: list[str]
    estimated_tools: list[str]
    priority: str

# 伪代码：很多上层库都提供类似接口
plan = llm.generate(
    prompt="帮我规划：写一篇关于 RAG 的技术博客",
    response_model=TaskPlan
)
```

schema-first 的重点不是某个具体库，而是一种工程组织方式：

- `Pydantic`、`Zod` 这类 schema 先成为系统里的"单一事实来源"
- 上层库再把它转成 JSON Schema、prompt 约束、重试策略或解析器
- 同一份 schema 可以复用于 API、数据库校验、前后端类型共享

也就是说，**schema-first libs 不是模型能力，而是应用层抽象**。它们底层可能调用原生 structured outputs，也可能只是 "prompt + parse + retry"。

### 方式四：在 Prompt 里约束格式（兜底方案）

```python
system = """你的输出必须严格遵循以下 JSON 格式，不要输出其他内容：

{
  "action": "search | answer | clarify",
  "content": "具体内容",
  "confidence": 0.0-1.0
}"""
```

这种方式最不稳定，但兼容性最好。当模型 API 不支持原生结构化输出时，它仍然是常见兜底方案。

### 一个判断框架

看到这些术语时，先问三个问题：

1. 我是在描述**模型原生能力**，还是**应用层封装方式**？
2. 我需要的是"输出可解析"，还是"生成过程本身被严格约束"？
3. 我是在追求**跨模型通用性**，还是某个平台上的**最强稳定性**？

用这三个问题去看，几个容易混淆的概念就分开了。

### Native Structured Outputs、schema-first libs、constrained decoding 到底是什么关系？

这三个概念经常被混着说，但它们其实处在不同层。

**第一层：schema-first libs**

这是应用层。你先定义 `Pydantic` / `Zod` 等 schema，再让上层库围绕它做 prompt 生成、参数校验、解析、重试和类型转换。

**第二层：Native Structured Outputs**

这是模型 API 层。平台直接接受 schema，并尽量让模型返回符合结构的结果。你感知到的是"这个模型原生支持结构化输出"。

**第三层：constrained decoding**

这是解码层。模型在逐 token 生成时，只允许输出符合某个 grammar、正则或 schema 的 token 路径。

比如生成 JSON 时，解码器会限制：

- 该出现 `{` 的时候不能先输出自然语言
- 某个字段只能输出枚举值中的一个
- 数组、对象、引号、逗号必须出现在合法位置

框架：

- Outlines https://github.com/dottxt-ai/outlines

很多人会把 Native Structured Outputs 和 constrained decoding 画等号，但更准确的说法是：

- `Native Structured Outputs` 是你看到的**产品能力**
- `constrained decoding` 是可能支撑它的**底层技术路线之一**

有些平台的原生结构化输出会使用接近 constrained decoding 的机制，有些则会叠加校验、重采样或其他内部策略。**作为 API 使用者，不要默认它们完全等价。**

最常见的工程组合是：

```text
Pydantic / Zod 定义 schema
    → schema-first 库把 schema 转成模型可用的约束
    → 模型 API 提供 Native Structured Outputs
    → 服务端可能在底层使用 constrained decoding 一类技术
    → 应用拿到已校验或可校验的结构化对象
```

### 工程上怎么选

- **单一平台、优先可靠性**：优先用 Native Structured Outputs
- **希望代码层有统一类型系统**：优先用 schema-first libs
- **本地模型 / 开源模型 / 极强格式约束**：重点看 constrained decoding
- **只有普通文本模型接口可用**：退回到 prompt 约束 + 校验 + 重试

它们不是互斥关系，而是经常叠加使用。真正要避免的误区只有一个：**把应用层的 schema-first、API 层的 native outputs、解码层的 constrained decoding 当成同一个东西。**

**在 Agent 里，Tool Calling 通常是最实用的 Structured Output 形式之一**——因为工具调用的参数和返回天然就是结构化数据，不需要再从自由文本里做二次提取。

---

## JSON Schema

JSON Schema 是描述"数据结构规则"的标准格式。它既可以用于 Tool Calling，也常常是 Native Structured Outputs 和 schema-first libs 背后的共同语言。

基本结构：

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "搜索关键词"
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 20,
      "default": 5
    },
    "format": {
      "type": "string",
      "enum": ["json", "markdown", "plain"],
      "description": "返回格式"
    },
    "filters": {
      "type": "array",
      "items": { "type": "string" },
      "description": "过滤条件列表"
    }
  },
  "required": ["query"]
}
```

核心字段：

| 字段              | 作用                                                      |
| ----------------- | --------------------------------------------------------- |
| `type`            | 数据类型：string, integer, number, boolean, array, object |
| `description`     | 告诉 LLM 这个字段的含义，直接影响 LLM 填参数的准确性      |
| `enum`            | 枚举值，限定 LLM 只能填这几个值                           |
| `required`        | 必填字段列表                                              |
| `minimum/maximum` | 数字范围约束                                              |
| `default`         | 默认值提示                                                |

**`description` 字段是最重要的**。LLM 完全靠 description 来理解参数的含义和用途。description 写得越准确，LLM 填参数越正确。

---

## 把这些能力串起来

在 Agent 场景下，这几个能力的配合关系：

```text
Role Prompting
    → 设定 Agent 的角色和职责边界

System Prompt
    → 定义 Agent 的行为规则和工具使用约束

Few-shot
    → 教 Agent 特定格式的工具调用方式

Chain-of-Thought
    → 让 Planner 在拆解任务前先推理清楚

JSON Schema
    → 定义工具参数结构，让 LLM 正确填参数

Structured Output
    → 确保 Agent 的输出（任务计划、分类结果等）可被下游代码解析

Tool Calling
    → 把结构化输出变成实际的系统动作（下一篇详细展开）
```

---

## 工程现场

场景：LLM 偶发性地输出格式不对的 JSON，导致 Agent 解析失败崩溃。

根本原因：LLM 的输出不是 100% 确定性的，即使加了 JSON Mode 也可能在边界情况下出错。

处理策略：

```python
import json
from json_repair import repair_json  # pip install json-repair

def safe_parse_json(content: str) -> dict | None:
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试修复后解析（处理常见的格式错误）
    try:
        repaired = repair_json(content)
        return json.loads(repaired)
    except:
        pass

    # 尝试提取 JSON 块
    import re
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None  # 解析失败，上层处理
```

加了容错层之后，JSON 解析失败率通常能降低 90% 以上。

---

## 小结

这几个能力在 Agent 里的优先级：

```text
必须掌握：
  System Prompt   — Agent 行为的基础控制
  Structured Output / JSON Schema — Tool Calling 的前提
  Tool Calling    — 下一篇的主题

需要理解：
  Role Prompting  — 提升 LLM 在特定任务上的表现
  Few-shot        — 格式复杂时效果明显
  Native Structured Outputs / schema-first / constrained decoding
                 — 分清结构化输出发生在哪一层

按需使用：
  Chain-of-Thought — 推理复杂任务（Planner 场景）时启用
```

掌握这些之后，进入 Tool Calling——把结构化输出变成真正的系统动作。
