# Cost Control：Agent 的成本控制

Agent 的成本问题和普通 LLM 应用不在一个量级。

普通 LLM 调用：1 次请求 = 1 次 LLM 调用。
Agent：1 次请求 = 5-15 次 LLM 调用 + 多次工具调用。

一个不加控制的 Agent，每次请求可能消耗 50K-200K token，成本是普通 LLM 应用的 10-50 倍。

---

## 成本结构分析

Agent 的成本由三部分组成：

```text
LLM 调用成本（占 80-90%）
    = Σ (prompt_tokens × 输入单价 + completion_tokens × 输出单价) × 每步调用次数

工具调用成本（占 5-15%）
    = 外部 API 调用费 + 向量搜索费 + 数据库查询费

基础设施成本（占 5-10%）
    = 服务器 + 向量数据库 + 存储
```

LLM 调用是大头，优化的重点也在这里。

---

## Token 消耗的来源

一次 Agent 执行的 Token 分布：

```text
System Prompt（每步重复发送）  → 占 30-50%
Messages History（累积增长）   → 占 20-40%
Tool Results（检索结果等）     → 占 10-20%
LLM 输出                     → 占 5-10%
```

最大的浪费来源：**System Prompt 和 Messages History 在每步都完整发送**。Agent 跑 10 步，就发送了 10 次完整的 System Prompt。

---

## 优化策略一：减少 Agent 步数

步数是成本的倍数放大器。减少一步 = 省掉一次完整的 LLM 调用（包括重复发送的 System Prompt 和 History）。

### 更好的 System Prompt

让 Agent 减少不必要的中间步骤：

```python
# 差：Agent 倾向于"先搜索再回答"，即使不需要搜索
"你是一个助手，使用工具来帮助用户。"

# 好：明确告诉 Agent 什么时候不需要工具
"""你是一个知识库问答助手。

效率原则：
- 如果你已有足够信息回答，直接回答，不要多余搜索
- 搜索时使用精准关键词，一次搜索尽量解决问题
- 最多搜索 2 次。超过 2 次说明问题可能超出知识库范围，直接告知用户"""
```

### 并行工具调用

如果需要多个独立的信息，引导 LLM 在一步里同时调用多个工具，而不是串行：

```python
# System Prompt 里引导并行调用
"""当你需要多个独立信息时，在同一步里同时调用多个工具，而不是一个一个调用。

例如：
- 需要查 A 的定义和 B 的定义 → 同时调用两次 search，不要先查 A 再查 B
"""
```

---

## 优化策略二：压缩 Token

### System Prompt 精简

```python
# 精简前（380 token）
"""你是一个专业的技术文档知识库问答助手。你的目标是准确地回答用户关于技术文档、产品功能、
API 接口等方面的问题。你需要使用 search_knowledge_base 工具来检索知识库中的相关信息，
然后基于检索结果给出准确的回答。在回答时，你需要引用信息来源。如果知识库中没有相关信息，
你应该诚实地告诉用户你没有找到相关内容，而不是编造答案。"""

# 精简后（120 token）
"""技术文档问答助手。
规则：1.必须先用search检索 2.基于检索结果回答并标注来源 3.找不到就说没有，不编造"""
```

System Prompt 每步重复发送，精简 260 token × 10 步 = 节省 2600 token。

### Messages History 压缩

```python
def compress_messages(messages: list, max_history_tokens: int = 3000) -> list:
    """压缩历史消息，保留关键信息"""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    recent = messages[-4:]  # 保留最近 2 轮（4 条消息）

    # 中间的消息做摘要
    middle = messages[len(system_msgs):-4]
    if middle and count_tokens(middle) > max_history_tokens:
        summary = summarize_messages(middle)  # LLM 摘要或规则摘要
        middle = [{"role": "system", "content": f"[历史摘要] {summary}"}]

    return system_msgs + middle + recent
```

### Tool Result 截断

检索结果往往很长，但 LLM 只需要关键部分：

```python
def truncate_tool_result(result: str, max_chars: int = 2000) -> str:
    """截断过长的工具返回值"""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + f"\n\n[结果已截断，共 {len(result)} 字符，仅显示前 {max_chars} 字符]"


# 更聪明的做法：让 LLM 提取关键信息
async def extract_key_info(tool_result: str, query: str, max_tokens: int = 500) -> str:
    """从冗长的工具返回中提取与问题相关的关键信息"""
    if count_tokens(tool_result) <= max_tokens:
        return tool_result

    prompt = f"""从以下内容中提取与问题相关的关键信息，控制在 {max_tokens} token 以内。

问题：{query}
内容：{tool_result[:5000]}

关键信息："""
    return await call_llm(prompt, model="gpt-4o-mini")  # 用便宜模型提取
```

---

## 优化策略三：模型路由

不同步骤用不同模型，大模型做决策，小模型做执行：

```python
class ModelRouter:
    """根据任务类型选择模型"""

    def __init__(self):
        self.routes = {
            "planning": "gpt-4o",        # 规划用大模型
            "tool_selection": "gpt-4o",   # 工具选择用大模型
            "simple_qa": "gpt-4o-mini",   # 简单问答用小模型
            "summarization": "gpt-4o-mini", # 摘要用小模型
            "classification": "gpt-4o-mini", # 分类用小模型
            "final_answer": "gpt-4o",     # 最终回答用大模型
        }

    def get_model(self, task_type: str) -> str:
        return self.routes.get(task_type, "gpt-4o-mini")


# 成本对比（GPT-4o vs GPT-4o-mini）
# GPT-4o:      $2.50/1M input,  $10.00/1M output
# GPT-4o-mini: $0.15/1M input,  $0.60/1M output
# 差距：输入 16x, 输出 16x
```

### 模型降级策略

当成本接近预算时，自动降级到更便宜的模型：

```python
class AdaptiveModelSelector:
    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self.spent = 0.0

    def select_model(self, task_type: str) -> str:
        remaining = self.budget - self.spent
        budget_ratio = remaining / self.budget

        if budget_ratio > 0.5:
            # 预算充足，用大模型
            return "gpt-4o"
        elif budget_ratio > 0.2:
            # 预算紧张，关键任务用大模型，其他用小模型
            if task_type in ("planning", "final_answer"):
                return "gpt-4o"
            return "gpt-4o-mini"
        else:
            # 预算接近用完，全部用小模型
            return "gpt-4o-mini"
```

---

## 优化策略四：缓存

### 语义缓存

相似问题的检索结果和答案可以缓存复用：

```python
class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self.cache: list[dict] = []  # 生产用 Redis + 向量索引

    async def get(self, query: str) -> str | None:
        query_embedding = await get_embedding(query)
        for item in self.cache:
            similarity = cosine_similarity(query_embedding, item["embedding"])
            if similarity >= self.threshold:
                return item["response"]
        return None

    async def set(self, query: str, response: str):
        embedding = await get_embedding(query)
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "response": response,
            "timestamp": datetime.now()
        })

# 在 Agent 入口使用
cache = SemanticCache(threshold=0.92)

async def agent_with_cache(user_input: str):
    cached = await cache.get(user_input)
    if cached:
        return cached  # 命中缓存，0 token 消耗

    result = await run_agent(user_input)
    await cache.set(user_input, result)
    return result
```

### Tool Result 缓存

同一工具 + 同参数的结果短期内不变，可以缓存：

```python
from functools import lru_cache
import hashlib

class ToolResultCache:
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, tool_name: str, args: dict) -> str | None:
        key = self._make_key(tool_name, args)
        item = self.cache.get(key)
        if item and (datetime.now() - item["time"]).total_seconds() < self.ttl:
            return item["result"]
        return None

    def set(self, tool_name: str, args: dict, result: str):
        key = self._make_key(tool_name, args)
        self.cache[key] = {"result": result, "time": datetime.now()}

    def _make_key(self, tool_name: str, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True)
        return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()
```

---

## 成本监控与报警

```python
# 成本计算
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
    return prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]


# 每日成本报告
async def daily_cost_report(traces: list[Trace]) -> dict:
    total_cost = sum(t.total_cost for t in traces)
    avg_cost = total_cost / len(traces) if traces else 0

    # 找出最贵的请求
    sorted_traces = sorted(traces, key=lambda t: t.total_cost, reverse=True)
    top_expensive = [
        {"input": t.user_input[:50], "cost": t.total_cost, "steps": t.step_count}
        for t in sorted_traces[:10]
    ]

    return {
        "date": datetime.now().date().isoformat(),
        "total_requests": len(traces),
        "total_cost_usd": round(total_cost, 2),
        "avg_cost_per_request": round(avg_cost, 4),
        "top_expensive_requests": top_expensive,
    }


# 成本报警
def check_cost_alert(daily_cost: float, budget: float):
    if daily_cost > budget * 0.8:
        send_alert(f"日成本 ${daily_cost:.2f} 已达预算 80%")
    if daily_cost > budget:
        send_alert(f"日成本 ${daily_cost:.2f} 已超预算！")
        # 可选：自动降级模型或限流
```

---

## 成本优化效果估算

假设 Agent 每天处理 1000 次请求，每次平均 8 步：

| 优化措施 | 节省比例 | 原理 |
|---------|---------|------|
| System Prompt 精简 | 15-25% | 每步少发 200+ token |
| Messages 压缩 | 10-20% | 历史消息摘要后 token 大幅减少 |
| Tool Result 截断 | 5-10% | 检索结果只保留关键部分 |
| 模型路由 | 30-50% | 简单步骤用 mini 模型 |
| 语义缓存（30% 命中率）| 25-30% | 命中的请求 0 token 消耗 |
| 减少步数 | 10-20% | 更好的 prompt 让 Agent 少走弯路 |

组合使用，总成本可以降低 50-70%。

---

## 工程现场

场景：Agent 的日成本突然翻了 3 倍，但请求量没有明显变化。

排查：

```python
# 1. 按请求维度查看 Token 分布
# 发现某类请求的平均 Token 从 15K 涨到 80K

# 2. 看具体 Trace
# 发现 Agent 在处理某类问题时陷入循环，反复调用搜索工具

# 3. 根因
# 知识库新增了一批文档，这些文档的内容和 Agent 的搜索关键词部分匹配但不够相关
# Agent 判断结果不满意 → 换关键词重搜 → 还是不满意 → 继续换 → 直到 max_steps
```

修复：
1. 加循环检测，连续搜索 3 次同类关键词后强制停止。
2. 在"结果不满意"的判断逻辑里加阈值："虽然不完美但可以给出部分回答"。
3. 给 `max_steps` 设一个更严格的值（从 15 降到 8）。

---

## 小结

Agent 成本控制的优先级：

```text
高 ROI（先做）：
    模型路由      — 简单任务用 mini，效果差距小但成本差 16x
    System Prompt 精简 — 改一次，每步都省
    max_steps 收紧 — 防止失控

中 ROI（第二轮）：
    语义缓存      — 需要一定基础设施，但高频场景收益大
    Messages 压缩  — 长对话场景收益明显
    Tool Result 截断 — 检索结果长时收益明显

持续做：
    成本监控 + 报警   — 第一时间发现异常
    定期分析 Top 高成本请求 — 针对性优化
```
