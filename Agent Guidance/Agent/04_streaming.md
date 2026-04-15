# Streaming：Agent 的实时输出

Agent 可能需要运行 10 到 30 秒，期间调用多个工具、多次推理。如果等所有步骤完成再返回，用户看到的是一段沉默后突然出现的文字。这种体验很差。

Streaming 解决两个问题：
1. 让用户看到 Agent 正在做什么（工具调用进度、中间推理）
2. 最终回答逐字输出，而不是等全部生成完再一次性显示

---

## LLM Streaming 基础

先理解 LLM 本身的流式输出：

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_llm():
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "介绍一下 RAG"}],
        stream=True  # 开启流式
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
```

流式返回的是一系列 chunk，每个 chunk 包含部分内容。LLM 生成一个 token 就推送一个 chunk，不用等全部生成完。

---

## Tool Calling 的流式解析

开启 streaming 后，tool call 参数也是分块返回的，需要拼接：

```python
async def stream_with_tool_calls(messages: list, tools: list):
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=True
    )

    # 用于拼接流式 tool call
    tool_calls_buffer = {}  # index -> {id, name, arguments}
    content_buffer = ""

    async for chunk in stream:
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        # 普通文本内容
        if delta.content:
            content_buffer += delta.content
            yield {"type": "content", "chunk": delta.content}

        # Tool call 分块到来
        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tc_chunk.id or "",
                        "name": "",
                        "arguments": ""
                    }

                if tc_chunk.id:
                    tool_calls_buffer[idx]["id"] = tc_chunk.id
                if tc_chunk.function.name:
                    tool_calls_buffer[idx]["name"] += tc_chunk.function.name
                if tc_chunk.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tc_chunk.function.arguments

        # 流结束
        if finish_reason == "tool_calls":
            # 所有 tool call 已拼接完整
            yield {
                "type": "tool_calls",
                "tool_calls": list(tool_calls_buffer.values())
            }
        elif finish_reason == "stop":
            yield {"type": "done", "content": content_buffer}
```

---

## Agent Loop 的流式实现

把流式输出整合进 Agent Loop：

```python
from typing import AsyncGenerator

async def run_agent_stream(
    user_input: str,
    tools: list,
    tool_executors: dict,
    max_steps: int = 10
) -> AsyncGenerator[dict, None]:

    messages = [
        {"role": "system", "content": "你是一个助手，使用工具来回答用户问题。"},
        {"role": "user", "content": user_input}
    ]

    for step in range(max_steps):
        # 通知前端：新的推理步骤开始
        yield {"type": "step_start", "step": step + 1}

        # 流式调用 LLM
        tool_calls_result = []
        final_content = ""

        async for event in stream_with_tool_calls(messages, tools):
            if event["type"] == "content":
                # 实时推送文字内容
                yield {"type": "token", "content": event["chunk"]}
                final_content += event["chunk"]

            elif event["type"] == "tool_calls":
                tool_calls_result = event["tool_calls"]

            elif event["type"] == "done":
                final_content = event["content"]

        # 没有工具调用，任务完成
        if not tool_calls_result:
            yield {"type": "done", "content": final_content}
            return

        # 通知前端：正在调用哪些工具
        for tc in tool_calls_result:
            yield {
                "type": "tool_start",
                "tool": tc["name"],
                "args": tc["arguments"]
            }

        # 构建 assistant message（需要模拟完整的 message 对象格式）
        assistant_message = build_assistant_message(tool_calls_result)
        messages.append(assistant_message)

        # 并发执行工具
        tasks = [
            execute_tool_safe(tc["name"], tc["arguments"], tc["id"], tool_executors)
            for tc in tool_calls_result
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            # 通知前端：工具执行完毕
            yield {
                "type": "tool_done",
                "tool": result["tool_name"],
                "result_preview": str(result["content"])[:100]
            }

            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": str(result["content"])
            })

    yield {"type": "max_steps_reached"}
```

---

## SSE：把流式输出推到前端

Server-Sent Events（SSE）是把 Agent 流式输出推给浏览器的标准方式：

```python
# FastAPI 示例
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/agent/stream")
async def agent_stream_endpoint(request: dict):
    user_input = request["message"]
    session_id = request.get("session_id", "default")

    async def event_generator():
        async for event in run_agent_stream(user_input, tools, tool_executors):
            # SSE 格式：data: {json}\n\n
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 结束信号
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲
        }
    )
```

前端消费：

```javascript
const response = await fetch('/agent/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ message: userInput })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') return;

        const event = JSON.parse(data);

        switch (event.type) {
            case 'step_start':
                showStepIndicator(event.step);
                break;
            case 'token':
                appendToResponse(event.content);  // 逐字追加
                break;
            case 'tool_start':
                showToolCallIndicator(event.tool, event.args);
                break;
            case 'tool_done':
                updateToolResult(event.tool, event.result_preview);
                break;
            case 'done':
                finalizeResponse();
                break;
        }
    }
}
```

---

## 流式输出的事件类型设计

Agent 的流式事件需要让前端知道发生了什么，同时不暴露过多内部细节：

```python
# 推荐的事件类型
AGENT_EVENTS = {
    # 进度类
    "step_start": {"step": int},           # 开始第 N 步推理
    "thinking": {"content": str},           # LLM 推理过程（可选展示）

    # 工具类
    "tool_start": {"tool": str, "args": str},   # 开始调用工具
    "tool_done": {"tool": str, "preview": str}, # 工具调用完成（结果预览）
    "tool_error": {"tool": str, "error": str},  # 工具调用失败

    # 输出类
    "token": {"content": str},              # 最终答案的流式 token
    "done": {"content": str},               # 任务完成

    # 错误类
    "error": {"message": str},              # Agent 级别的错误
    "max_steps": {},                         # 达到步数限制
}
```

---

## 流式中断与恢复

用户可能在 Agent 运行中途关闭连接。需要处理这种情况：

```python
import asyncio

async def run_agent_stream_with_cancel(user_input: str, cancel_event: asyncio.Event):
    async for event in run_agent_stream(user_input, tools, tool_executors):
        # 检查是否被取消
        if cancel_event.is_set():
            yield {"type": "cancelled"}
            return

        yield event

# FastAPI 端点：检测客户端断开
@app.post("/agent/stream")
async def agent_stream_endpoint(request: Request, body: dict):
    cancel_event = asyncio.Event()

    async def event_generator():
        try:
            async for event in run_agent_stream_with_cancel(body["message"], cancel_event):
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 工程现场

场景：Nginx 代理后，SSE 流式输出出现"积压"——用户不是实时看到每个 token，而是每隔几秒收到一批。

原因：Nginx 默认开启了响应缓冲，把小的 chunk 积攒到一定大小再发出去。

修复：在响应头加 `X-Accel-Buffering: no`，或者在 Nginx 配置里对 SSE 路径关闭缓冲：

```nginx
location /agent/stream {
    proxy_pass http://backend;
    proxy_buffering off;          # 关闭代理缓冲
    proxy_cache off;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding on;
}
```

另一个常见问题：SSE 连接在 60 秒后被负载均衡器强制断开。修复方式是定期发送心跳：

```python
async def event_generator_with_heartbeat():
    heartbeat_task = asyncio.create_task(send_heartbeats())
    try:
        async for event in run_agent_stream(user_input, tools, tool_executors):
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        heartbeat_task.cancel()

async def send_heartbeats():
    while True:
        await asyncio.sleep(30)
        yield ": heartbeat\n\n"  # SSE 注释行，不触发前端事件处理
```

---

## 小结

Agent Streaming 不是简单地开一个 `stream=True`，而是需要：

1. **流式解析 tool call**：tool call 参数分块到来，需要在客户端拼接完整
2. **设计事件类型**：让前端知道 Agent 在哪个阶段（推理、工具调用、生成答案）
3. **SSE 推送**：标准格式、禁用代理缓冲、定期心跳
4. **取消机制**：用户断开时能及时停止 Agent 执行，避免浪费 token

到这里，Agent 核心引擎的四个模块（Tool Calling → Agent Loop → Memory → Streaming）已经完整。下一步进入阶段二：把你已有的 RAG 链路封装成 Agent 工具。
