# Observability：Agent 的可观测性

Agent 上线后，最大的挑战不是"能不能跑"，而是"出了问题能不能定位"。

普通后端服务的可观测性已经成熟：日志、指标、链路追踪。Agent 在此基础上多了几个独特的问题：

- LLM 调用是黑盒，你不知道模型为什么做出某个决策
- Agent Loop 多步执行，需要追踪每一步的输入输出
- Tool 调用涉及外部系统，失败原因可能在任何环节
- Token 消耗直接等于成本，需要精确统计

---

## Agent 可观测性的三层结构

```text
Layer 1：Trace（链路追踪）
    一次 Agent 执行的完整链路：从用户输入到最终输出

Layer 2：Span（步骤追踪）
    链路内的每一步：LLM 调用、Tool 执行、检索、生成

Layer 3：Metrics（指标统计）
    聚合数据：延迟分布、成功率、Token 消耗、成本
```

---

## Layer 1：Trace——一次完整的 Agent 执行

一个 Trace 记录从用户输入到最终输出的全过程：

```python
import uuid
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str = ""
    final_output: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    spans: list = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    status: str = "running"  # running | success | error
    error: str = None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "user_input": self.user_input,
            "final_output": self.final_output[:200],
            "duration_ms": (self.end_time - self.start_time).total_seconds() * 1000 if self.end_time else None,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "status": self.status,
            "step_count": len([s for s in self.spans if s.span_type == "llm_call"]),
            "tool_calls": len([s for s in self.spans if s.span_type == "tool_call"]),
        }
```

---

## Layer 2：Span——每一步的详细记录

```python
@dataclass
class Span:
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    parent_span_id: str = None
    span_type: str = ""  # llm_call | tool_call | retrieval | generation
    name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    status: str = "running"
    error: str = None

# LLM 调用的 Span
llm_span = Span(
    span_type="llm_call",
    name="gpt-4o",
    input_data={
        "messages_count": 5,
        "messages_preview": messages[-1]["content"][:100],
        "tools_count": 3
    },
    output_data={
        "has_tool_calls": True,
        "tool_calls": [{"name": "search", "args": {"query": "rerank"}}],
        "tokens": {"prompt": 1200, "completion": 85}
    }
)

# Tool 调用的 Span
tool_span = Span(
    span_type="tool_call",
    name="search_knowledge_base",
    input_data={"query": "rerank 原理", "top_k": 5},
    output_data={
        "status": "success",
        "chunks_found": 3,
        "result_preview": "rerank 是用来提升检索精度的..."[:100]
    },
    metadata={"latency_ms": 230}
)
```

---

## 手写 Trace 收集器

不依赖第三方服务，先用最简单的方式把 Trace 记录下来：

```python
import json
from pathlib import Path

class TraceCollector:
    def __init__(self, storage_dir: str = "./traces"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._current_trace: Trace = None

    def start_trace(self, user_input: str) -> Trace:
        self._current_trace = Trace(user_input=user_input)
        return self._current_trace

    def add_span(self, span: Span):
        span.trace_id = self._current_trace.trace_id
        self._current_trace.spans.append(span)

    def end_trace(self, output: str, status: str = "success", error: str = None):
        trace = self._current_trace
        trace.final_output = output
        trace.end_time = datetime.now()
        trace.status = status
        trace.error = error

        # 汇总 token
        for span in trace.spans:
            if span.span_type == "llm_call" and "tokens" in span.output_data:
                tokens = span.output_data["tokens"]
                trace.total_tokens += tokens.get("prompt", 0) + tokens.get("completion", 0)

        # 持久化
        filepath = self.storage_dir / f"{trace.trace_id}.json"
        filepath.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))

        self._current_trace = None
        return trace


# 集成到 Agent Loop
collector = TraceCollector()

async def run_agent_with_tracing(user_input: str) -> str:
    trace = collector.start_trace(user_input)

    messages = [{"role": "user", "content": user_input}]

    for step in range(MAX_STEPS):
        # LLM Span
        llm_span = Span(span_type="llm_call", name="gpt-4o")
        response = await call_llm(messages, tools)
        llm_span.end_time = datetime.now()
        llm_span.output_data = {
            "has_tool_calls": bool(response.choices[0].message.tool_calls),
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens
            }
        }
        collector.add_span(llm_span)

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            collector.end_trace(message.content, "success")
            return message.content

        # Tool Spans
        for tc in message.tool_calls:
            tool_span = Span(span_type="tool_call", name=tc.function.name)
            try:
                result = await execute_tool(tc)
                tool_span.status = "success"
                tool_span.output_data = {"result_preview": str(result)[:200]}
            except Exception as e:
                tool_span.status = "error"
                tool_span.error = str(e)
            tool_span.end_time = datetime.now()
            collector.add_span(tool_span)

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    collector.end_trace("max steps reached", "error")
    return "达到最大步数"
```

---

## 接入 LangSmith

LangSmith 是 LangChain 官方的可观测性平台，对 LangGraph Agent 支持最好。

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-agent"

# 设置环境变量后，LangGraph 自动上报所有 Trace
# 不需要改代码
app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="test")]})
# Trace 自动出现在 LangSmith dashboard
```

LangSmith 提供的能力：

```text
- 完整的 Trace 可视化（每步的输入输出）
- LLM 调用的 prompt/completion 详情
- Token 消耗和延迟统计
- 回归测试（用历史 Trace 做自动化测试）
- 人工标注（标记 Trace 质量）
```

---

## 接入 OpenTelemetry

如果已有 OpenTelemetry 基础设施（Jaeger / Grafana Tempo），可以直接对接：

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agent-service")

async def run_agent_with_otel(user_input: str):
    with tracer.start_as_current_span("agent_run") as root_span:
        root_span.set_attribute("user_input", user_input[:200])

        for step in range(MAX_STEPS):
            with tracer.start_as_current_span("llm_call") as llm_span:
                response = await call_llm(messages, tools)
                llm_span.set_attribute("model", "gpt-4o")
                llm_span.set_attribute("prompt_tokens", response.usage.prompt_tokens)
                llm_span.set_attribute("completion_tokens", response.usage.completion_tokens)

            if message.tool_calls:
                for tc in message.tool_calls:
                    with tracer.start_as_current_span("tool_call") as tool_span:
                        tool_span.set_attribute("tool_name", tc.function.name)
                        result = await execute_tool(tc)
                        tool_span.set_attribute("status", "success")
```

OpenTelemetry 的优势：接入已有的后端监控体系（Grafana、Datadog、New Relic），不需要额外的 LLM 专用平台。

---

## 核心监控指标

### 请求级指标

| 指标 | 计算方式 | 报警阈值参考 |
|------|---------|------------|
| 端到端延迟 | trace.end_time - trace.start_time | P95 > 30s |
| Agent 步数 | count(llm_call spans) | > 8 步 |
| Token 消耗 | sum(prompt_tokens + completion_tokens) | > 50K/次 |
| 任务成功率 | success_traces / total_traces | < 90% |
| 工具错误率 | error_tool_spans / total_tool_spans | > 10% |

### 聚合指标（Dashboard）

```python
# 每日统计
daily_stats = {
    "total_requests": count_traces(today),
    "success_rate": success_traces / total_traces,
    "avg_latency_ms": avg(trace_durations),
    "p95_latency_ms": percentile(trace_durations, 95),
    "total_tokens": sum(trace_tokens),
    "total_cost_usd": sum(trace_costs),
    "avg_steps_per_request": avg(trace_steps),
    "tool_error_rate": error_tool_calls / total_tool_calls,
    "top_tools_by_usage": counter(tool_names).most_common(5),
    "top_errors": counter(error_messages).most_common(5),
}
```

### Agent 特有指标

```text
循环检测率 — Agent 陷入循环的比例
幻觉率 — 答案超出检索结果的比例（需要人工标注或自动评估）
工具选择准确率 — LLM 选对工具的比例
检索触发率 — 需要检索时确实触发了检索的比例
```

---

## 日志规范

Agent 日志需要比普通服务更结构化：

```python
import structlog

logger = structlog.get_logger()

# 每步记录
logger.info("agent_step",
    trace_id=trace.trace_id,
    step=step,
    action="llm_call",
    model="gpt-4o",
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
    has_tool_calls=bool(message.tool_calls),
    latency_ms=elapsed
)

logger.info("tool_execution",
    trace_id=trace.trace_id,
    step=step,
    tool_name=tc.function.name,
    tool_args=json.loads(tc.function.arguments),
    status="success",
    latency_ms=tool_elapsed,
    result_preview=str(result)[:100]
)

# 异常记录
logger.error("agent_error",
    trace_id=trace.trace_id,
    step=step,
    error_type=type(e).__name__,
    error_message=str(e),
    last_action=last_action,
    messages_count=len(messages)
)
```

---

## 工程现场

场景：Agent 在生产中偶尔出现延迟飙到 60 秒以上，但日志看不出具体原因。

原因：只记录了每步的开始和结束，没有记录 LLM 调用和 Tool 调用的耗时分布。

修复：给每个 Span 加精确的 latency 统计，然后按步骤拆分延迟：

```python
# Trace 完成后，输出延迟分解
def print_latency_breakdown(trace: Trace):
    total = (trace.end_time - trace.start_time).total_seconds()
    llm_time = sum(
        (s.end_time - s.start_time).total_seconds()
        for s in trace.spans if s.span_type == "llm_call"
    )
    tool_time = sum(
        (s.end_time - s.start_time).total_seconds()
        for s in trace.spans if s.span_type == "tool_call"
    )
    overhead = total - llm_time - tool_time

    print(f"总耗时: {total:.1f}s")
    print(f"  LLM 调用: {llm_time:.1f}s ({llm_time/total*100:.0f}%)")
    print(f"  Tool 执行: {tool_time:.1f}s ({tool_time/total*100:.0f}%)")
    print(f"  框架开销: {overhead:.1f}s ({overhead/total*100:.0f}%)")
```

通常发现：80% 的延迟来自 LLM 调用，优化方向是减少 Agent 步数或用更小的模型处理简单步骤。

---

## 小结

Agent 可观测性的最小可行方案：

```text
第一步：手写 TraceCollector（上面的代码），记录到文件
    → 能定位"哪一步出了问题"

第二步：加结构化日志（structlog）
    → 能在日志搜索里快速筛选异常

第三步：接入 LangSmith 或 OpenTelemetry
    → 有可视化 Dashboard 和报警

第四步：加聚合指标和报警
    → 延迟、成功率、成本的自动监控
```

下一篇讲 Evaluation：如何系统评估 Agent 的效果，而不只是靠感觉。
