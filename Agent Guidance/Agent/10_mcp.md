# MCP 与 Skill：工具标准化与能力复用

Agent 的工具越来越多时，两个问题逐渐突出：

1. **接口不统一**：每个工具格式不同，每接入一个新工具就要写适配代码
2. **缺少复用机制**：Agent 反复解决同类问题，但每次从头推理，没有积累

MCP 解决第一个问题——标准化工具接口。Skill 解决第二个问题——让 Agent 积累和复用能力。

---

## Part 1：MCP（Model Context Protocol）

### 为什么需要 MCP

Anthropic 在 2024 年底发布的开放协议，目标是**标准化 LLM 应用与外部工具/数据源之间的通信方式**。

类比：MCP 之于 Agent 工具，相当于 USB 之于外设——统一接口标准，即插即用。

MCP 之前：

```text
工具 A（REST API）→ 写 HTTP 调用 + 参数适配 + 返回值解析
工具 B（Python SDK）→ 写 SDK 调用 + 错误处理
工具 C（CLI 命令）→ 写 subprocess 调用 + 输出解析
```

N 个工具 × M 个 Agent 框架 = N × M 种适配代码。

MCP 之后：N + M。

```text
工具 A ──┐                  ┌── Agent 框架 A
工具 B ──┼── MCP 协议 ──┼── Agent 框架 B
工具 C ──┘                  └── Agent 框架 C
```

---

### MCP 架构

```text
┌──────────────┐          ┌──────────────┐
│  MCP Client  │ ◄──────▶ │  MCP Server  │
│ （Agent 侧）  │   JSON-RPC   │ （工具侧）  │
└──────────────┘          └──────────────┘
```

**MCP Server**：暴露工具能力的服务。每个 Server 提供一组工具。

**MCP Client**：Agent 侧的连接器，发现并调用 MCP Server 上的工具。

**通信协议**：JSON-RPC 2.0。

两种传输方式：

```text
stdio  — Server 作为子进程运行，通过标准输入输出通信（本地开发常用）
SSE    — Server 作为独立 HTTP 服务，通过 Server-Sent Events 通信（生产部署）
```

---

### MCP Server 实现

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("knowledge-base-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge_base",
            description="在知识库中搜索相关文档。适用于回答产品功能和技术文档相关问题。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_knowledge_base":
        results = await do_search(arguments["query"], arguments.get("top_k", 5))
        return [TextContent(type="text", text=format_results(results))]
    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

### MCP Client 接入

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def connect_and_use():
    server_params = StdioServerParameters(
        command="python",
        args=["knowledge_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 发现工具
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"Tool: {tool.name} - {tool.description}")

            # 调用工具
            result = await session.call_tool(
                "search_knowledge_base",
                arguments={"query": "rerank 原理", "top_k": 5}
            )
            print(result.content[0].text)
```

---

### MCP 工具接入 Agent Loop

```python
async def mcp_tools_to_openai_format(session: ClientSession) -> list[dict]:
    """MCP 工具列表 → OpenAI tool calling 格式"""
    mcp_tools = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        for tool in mcp_tools.tools
    ]


async def run_agent_with_mcp(user_input: str, session: ClientSession):
    tools = await mcp_tools_to_openai_format(session)
    messages = [{"role": "user", "content": user_input}]

    for step in range(MAX_STEPS):
        response = await call_llm(messages, tools)
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = await session.call_tool(tc.function.name, arguments=args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result.content[0].text
            })

    return "达到最大步数"
```

---

### MCP 的三种能力

MCP Server 不只提供工具：

```text
Tools       — 可调用的功能（搜索、查询、执行操作）
Resources   — 可读取的数据（文件内容、数据库记录、配置）
Prompts     — 可复用的 prompt 模板
```

```python
# Resources：只读数据
@server.list_resources()
async def list_resources():
    return [
        Resource(uri="knowledge://product-docs", name="产品文档", description="完整产品文档")
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "knowledge://product-docs":
        return load_product_docs()

# Prompts：可复用模板
@server.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="rag-answer",
            description="基于检索结果生成答案的 prompt 模板",
            arguments=[
                PromptArgument(name="query", required=True),
                PromptArgument(name="context", required=True)
            ]
        )
    ]
```

---

### 多 MCP Server 组合

```python
config = {
    "mcpServers": {
        "knowledge-base": {"command": "python", "args": ["servers/knowledge_server.py"]},
        "database": {"command": "python", "args": ["servers/db_server.py"]},
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}
    }
}
```

已有的社区 MCP Server 生态：

| Server | 功能 |
|--------|------|
| `server-filesystem` | 文件读写 |
| `server-github` | GitHub 操作 |
| `server-postgres` | PostgreSQL 查询 |
| `server-slack` | Slack 消息 |
| `server-brave-search` | 网页搜索 |

---

### 把已有 RAG 封装为 MCP Server

你的 RAG 系统直接封装为 MCP Server，Claude Desktop、Cursor、任何 MCP Client 都可以调用：

```python
server = Server("rag-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search",
            description="语义搜索知识库",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "rerank": {"type": "boolean", "default": True}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_by_metadata",
            description="按元数据过滤搜索（部门、文档类型、时间范围）",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "department": {"type": "string", "enum": ["engineering", "product", "support"]},
                    "doc_type": {"type": "string", "enum": ["api_doc", "guide", "faq"]}
                },
                "required": ["query"]
            }
        )
    ]
```

---

## Part 2：Skill（Agent 能力复用）

### 什么是 Skill

Skill 是 Agent 的**可复用能力单元**。工具（Tool）是原子操作，Skill 是工具 + 流程 + 经验的组合。

```text
Tool  = 一次 API 调用（搜索、写文件）
Skill = 一系列 Tool 调用 + 推理逻辑 + 最佳实践
```

举例：

```text
Tool：search_knowledge_base（一次检索）
Skill："回答用户的产品功能问题"
      = search_knowledge_base（检索）
      + 判断结果是否充分
      + 必要时换关键词重试
      + 按标准格式输出带引用的答案
```

Skill 的概念来自 **Voyager**（2023）——在 Minecraft 里，Agent 学会了"建造房子"这个 Skill，以后遇到类似需求直接复用，不需要从挖第一块泥土开始推理。

---

### 理论基础：Voyager 的 Skill Library

论文：*Voyager: An Open-Ended Embodied Agent with Large Language Models*（Wang et al., 2023）

Voyager 的核心创新：Agent 每成功完成一个任务，就把解决方案存为一个 Skill，形成不断增长的 Skill Library。

```text
Skill Library
├── mine_wood()        — 砍树获取木材
├── craft_planks()     — 木材合成木板
├── build_shelter()    — 建造庇护所（调用 mine_wood + craft_planks + 放置方块）
└── ...

新任务到来时：
1. 在 Skill Library 里检索相关 Skill
2. 如果找到，直接调用已有 Skill
3. 如果没有，从头推理 → 执行 → 成功后存为新 Skill
```

这个思路直接适用于工程场景：把 Agent 成功执行的 workflow 存下来，下次遇到类似问题直接调用。

---

### 工程实现：Skill 的三种形态

#### 形态一：Prompt Skill（最轻量）

把成功的 prompt + 工具组合存为模板：

```python
class PromptSkill:
    def __init__(self, name: str, description: str, system_prompt: str, tools: list[str]):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools  # 需要的工具名列表

# 示例
product_qa_skill = PromptSkill(
    name="product_qa",
    description="回答用户关于产品功能的问题，基于知识库检索",
    system_prompt="""你是产品问答助手。

流程：
1. 先调用 search_knowledge_base 检索相关信息
2. 如果结果不足，换关键词再检索一次（最多 2 次）
3. 基于检索结果回答，标注来源
4. 如果确实找不到，回复"知识库中暂无相关信息"

输出要求：
- 简洁直接，不超过 200 字
- 附带来源引用""",
    tools=["search_knowledge_base"]
)
```

#### 形态二：Code Skill（Voyager 风格）

把成功的执行逻辑存为可调用的代码：

```python
import json
from pathlib import Path

class SkillLibrary:
    def __init__(self, storage_dir: str = "./skills"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.skills: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        for f in self.storage_dir.glob("*.json"):
            skill = json.loads(f.read_text())
            self.skills[skill["name"]] = skill

    def save_skill(self, name: str, description: str, code: str, dependencies: list[str] = None):
        """保存一个新 Skill"""
        skill = {
            "name": name,
            "description": description,
            "code": code,
            "dependencies": dependencies or [],
            "success_count": 0,
            "created_at": datetime.now().isoformat()
        }
        self.skills[name] = skill
        (self.storage_dir / f"{name}.json").write_text(json.dumps(skill, ensure_ascii=False, indent=2))

    async def search_skills(self, query: str, top_k: int = 3) -> list[dict]:
        """根据任务描述检索相关 Skill"""
        query_embedding = await get_embedding(query)
        scored = []
        for skill in self.skills.values():
            skill_embedding = await get_embedding(skill["description"])
            score = cosine_similarity(query_embedding, skill_embedding)
            scored.append((skill, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:top_k]]

    def record_success(self, name: str):
        """记录 Skill 成功执行"""
        if name in self.skills:
            self.skills[name]["success_count"] += 1
            self._save(name)


# 使用
library = SkillLibrary()

# Agent 成功完成了"分析 CSV 数据"任务后，存为 Skill
library.save_skill(
    name="analyze_csv",
    description="读取 CSV 文件，计算统计指标，生成分析报告",
    code="""
async def analyze_csv(file_path: str) -> str:
    data = await read_file(file_path)
    df = pd.read_csv(StringIO(data))
    stats = df.describe().to_string()
    report = await call_llm(f"基于以下统计数据生成简要分析报告：\\n{stats}")
    return report
""",
    dependencies=["read_file"]
)
```

#### 形态三：Workflow Skill（LangGraph 子图）

把验证过的 LangGraph 子图存为 Skill：

```python
# 一个经过验证的 RAG QA workflow，封装为可复用的 Skill
def create_rag_qa_skill():
    """创建 RAG 问答 Skill（LangGraph 子图）"""
    class RAGQAState(TypedDict):
        query: str
        context: str
        answer: str
        is_sufficient: bool

    def retrieve(state):
        context = do_retrieval(state["query"])
        return {"context": context}

    def check_sufficiency(state):
        is_sufficient = judge_context_quality(state["query"], state["context"])
        return {"is_sufficient": is_sufficient}

    def generate(state):
        answer = do_generation(state["query"], state["context"])
        return {"answer": answer}

    def retry_retrieval(state):
        new_query = rewrite_query(state["query"])
        context = do_retrieval(new_query)
        return {"context": context, "query": new_query}

    graph = StateGraph(RAGQAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check", check_sufficiency)
    graph.add_node("generate", generate)
    graph.add_node("retry", retry_retrieval)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges("check", lambda s: "generate" if s["is_sufficient"] else "retry")
    graph.add_edge("retry", "check")
    graph.add_edge("generate", END)

    return graph.compile()

# 在主 Agent 里作为一个 Skill 调用
rag_qa_skill = create_rag_qa_skill()
result = rag_qa_skill.invoke({"query": "rerank 的原理是什么"})
```

---

### Skill 的发现与选择

Agent 在运行时需要判断是否有现成 Skill 可用：

```python
async def run_agent_with_skills(user_input: str, skill_library: SkillLibrary):
    # Step 1：检索相关 Skill
    relevant_skills = await skill_library.search_skills(user_input, top_k=3)

    if relevant_skills and relevant_skills[0]["success_count"] > 3:
        # 高置信度：直接使用 Skill
        skill = relevant_skills[0]
        result = await execute_skill(skill, user_input)
        skill_library.record_success(skill["name"])
        return result

    elif relevant_skills:
        # 中置信度：把 Skill 作为参考，但仍由 Agent 自主推理
        skill_hints = format_skill_hints(relevant_skills)
        messages = [
            {"role": "system", "content": f"可参考的已有解决方案：\n{skill_hints}"},
            {"role": "user", "content": user_input}
        ]
        result = await run_agent(messages)

        # 任务成功后，判断是否需要更新 Skill
        await maybe_update_skill(skill_library, user_input, result)
        return result

    else:
        # 无相关 Skill：从头推理
        result = await run_agent([{"role": "user", "content": user_input}])

        # 任务成功后，提炼为新 Skill
        await maybe_save_new_skill(skill_library, user_input, result)
        return result
```

---

### MCP Prompts 作为 Skill 的分发机制

MCP 的 Prompts 能力天然适合做 Skill 的标准化分发：

```python
@server.list_prompts()
async def list_prompts():
    # 每个 Skill 对外暴露为一个 MCP Prompt
    return [
        Prompt(
            name="product-qa",
            description="回答产品功能问题（含知识库检索和引用）",
            arguments=[
                PromptArgument(name="question", description="用户问题", required=True)
            ]
        ),
        Prompt(
            name="csv-analysis",
            description="分析 CSV 数据文件并生成报告",
            arguments=[
                PromptArgument(name="file_path", description="CSV 文件路径", required=True)
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "product-qa":
        return GetPromptResult(messages=[{
            "role": "user",
            "content": f"""使用以下流程回答问题：
1. 调用 search_knowledge_base 检索
2. 判断结果是否充分，不够则换关键词重试
3. 基于检索结果回答，附带来源引用

问题：{arguments['question']}"""
        }])
```

这样 Skill 的定义和使用完全分离：MCP Server 维护 Skill，任何 MCP Client 可以发现和使用。

---

### MCP 与直接 Tool Calling 的对比

| 维度 | 直接 Tool Calling | MCP |
|------|------------------|-----|
| 耦合度 | Agent 代码直接调用工具 | Agent 通过协议调用，工具可独立部署 |
| 复用性 | 换框架需重写 | 一次实现，任何 MCP Client 可用 |
| 发现 | 硬编码工具列表 | 动态发现 Server 上的工具 |
| 部署 | 工具和 Agent 同进程 | Server 可独立进程/独立机器 |
| 开发速度 | 快（直接调用） | 需要多写 Server 层 |

小项目直接 Tool Calling 更快。跨项目复用、多团队协作的场景，MCP 价值更大。

---

## 工程现场

**MCP 问题**：MCP Server（stdio 模式）在高并发下频繁"卡死"。

原因：stdio 是单通道，一次只能处理一个请求。

修复：生产环境切 SSE 传输，Server 内部异步化：

```python
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route

sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/messages", endpoint=sse.handle_post_message, methods=["POST"])
])
```

**Skill 问题**：Skill Library 里存了太多低质量 Skill，检索时经常返回不相关的 Skill，干扰 Agent 判断。

修复：加入 Skill 淘汰机制：

```python
def cleanup_skills(library: SkillLibrary, min_success_rate: float = 0.5):
    """淘汰成功率低的 Skill"""
    for name, skill in list(library.skills.items()):
        total = skill.get("call_count", 0)
        success = skill.get("success_count", 0)
        if total > 5 and (success / total) < min_success_rate:
            library.remove_skill(name)
```

---

## 小结

```text
MCP 解决：工具接口标准化
    → 一次实现，任何 MCP Client 可用
    → 已有 RAG 系统封装为 MCP Server 后立即可被外部 Agent 调用

Skill 解决：Agent 能力复用
    → Prompt Skill：最轻量，prompt 模板复用
    → Code Skill：Voyager 风格，成功流程存为代码
    → Workflow Skill：LangGraph 子图，最强复用

MCP + Skill 配合：
    → MCP Prompts 作为 Skill 的标准化分发机制
    → MCP Tools 作为 Skill 的原子能力来源
    → Skill Library 作为 Agent 的能力积累层
```

阶段三到这里结束。下一步进入阶段四：Observability、Evaluation、Guardrails、Cost Control——把 Agent 推上生产。
