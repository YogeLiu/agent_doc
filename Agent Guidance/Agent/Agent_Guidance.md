# Agent 工程指南：从 RAG 工程师到 Agent 工程师

## 1. 先定认知，不先选框架

Agent 系统成败通常不在模型，而在工程结构：

1. Agent 能不能稳定完成多步任务，而不是在中途卡死或乱跑。
2. 工具调用能不能可靠执行，失败后能不能正确恢复。
3. 延迟、成本和可观测性能不能在生产环境长期维持。

核心认知转变只有一个：

> Agent 不是一次 LLM 调用，是一个有状态的循环系统。
> 后端经验（状态机、异步、错误处理、服务拆分）在这里全部适用。

---

## 2. Agent 系统完整结构

```text
User Input
    │
    ▼
Agent Loop
    │
    ├── Planner        （任务拆解）
    ├── Tool System    （执行动作）
    ├── Memory         （状态管理）
    └── RAG            （知识检索）
    │
    ▼
Orchestration          （多 Agent 协作）
    │
    ▼
Observability + Eval   （监控与评估）
```

核心公式：

```text
Agent = LLM + Tool System + Memory + Runtime + RAG + Orchestration
```

---

## 3. 学习路线（基于已有 RAG 基础）

### 阶段一：Agent 核心引擎（3 周）

目标：不依赖框架，手写一个可运行的 Agent。

```text
Tool Calling
    │
    ▼
Agent Loop（ReAct）
    │
    ▼
Memory 系统
    │
    ▼
Streaming 输出
```

**为什么先手写，不先用 LangChain：**
框架封装了细节。手写一次 Loop，才能理解框架在解决什么问题。

---

### 阶段二：RAG Agent（1.5 周，已有优势）

目标：把已有的 RAG 链路封装成 Agent 工具。

```text
Retriever as Tool
    │
    ▼
Agentic RAG（Agent 决定何时检索）
    │
    ▼
Self-RAG（Agent 对检索结果自我评估后决定是否重检索）
```

RAG 已经是最难的模块，这个阶段是迁移而非新学。

---

### 阶段三：编排框架（3 周）

目标：用 LangGraph 构建复杂多步 Agent 系统。

```text
LangGraph 基础（StateGraph, Node, Edge）
    │
    ▼
LangGraph 进阶（并发节点, 子图, Human-in-the-loop）
    │
    ▼
Multi-Agent（Supervisor 模式, Handoff）
    │
    ▼
MCP 协议（工具标准化）
```

---

### 阶段四：生产系统（持续）

目标：Agent 能上生产，可监控，可评估，可控制。

```text
Observability（LangSmith / OpenTelemetry）
    │
    ▼
Evaluation（Agent 任务成功率, 工具调用准确率）
    │
    ▼
Guardrails（输入过滤, 输出校验, 权限控制）
    │
    ▼
Cost Control（Token 预算, 缓存, 模型降级）
```

---

## 4. 阶段一重点：Agent Loop 原理

Agent 的核心是一个循环：

```text
while not done:
    1. 调用 LLM，输入当前 messages
    2. 解析 LLM 输出（是 tool call 还是 final answer）
    3. 如果是 tool call → 执行工具 → 把结果追加到 messages
    4. 如果是 final answer → 返回结果，退出循环
```

三个工程问题必须处理：

1. **终止条件**：最大步数限制，防止无限循环。
2. **工具失败处理**：工具报错时 Agent 如何恢复，而不是崩溃。
3. **并行工具调用**：LLM 返回多个 tool call 时，并发执行提升性能。

来自论文 ReAct（2022）：Reason + Act 交替进行，每次 Act 后把 Observe 结果送回 LLM。

---

## 5. 阶段三重点：LangGraph 状态机模型

LangGraph 把 Agent 建模为状态图：

```text
State（共享状态）
    │
Node（处理函数）── Edge（转移条件）──▶ Node
    │
Conditional Edge（根据 State 动态路由）
```

与传统 DAG 的区别：LangGraph 支持循环，Agent Loop 天然就是一个有环图。

核心设计原则：

- State 是单一数据源，所有 Node 读写同一个 State。
- Node 是纯函数，输入 State，输出 State 的变更。
- Edge 决定下一步去哪，Conditional Edge 让 Agent 能动态决策。

---

## 6. Multi-Agent 核心模式

```text
Supervisor 模式：
    Supervisor Agent
        ├── 分配任务给 Sub-Agent A
        ├── 分配任务给 Sub-Agent B
        └── 汇总结果，决定下一步

Swarm 模式：
    Agent A ──handoff──▶ Agent B ──handoff──▶ Agent C
    （每个 Agent 专注一件事，完成后移交下一个）
```

选择依据：

- 任务有明确分工且顺序固定 → Swarm。
- 任务动态分配，需要协调 → Supervisor。

---

## 7. 评估体系

Agent 评估比 RAG 更难，因为任务是多步骤的。

| 评估维度 | 指标 |
|---------|------|
| 任务完成 | Task Success Rate |
| 工具调用 | Tool Call Accuracy, Tool Error Rate |
| 检索质量 | 复用 RAG 评估指标 |
| 效率 | 平均步数, Token 消耗 |
| 安全 | Prompt Injection 检出率 |

评估难点：中间步骤正确但最终结果错误，或最终结果正确但路径低效，都需要分开统计。

---

## 8. Demo 容易，落地难：如何找到生产入口

这是从 Agent 学习者到 Agent 工程师最关键的一跳。

**根本原因**：Demo 解决的是"LLM 能不能做到"，生产解决的是"这件事值不值得用 Agent 做，以及如何做可靠"。

### 8.1 识别 Agent 适合的任务类型

适合用 Agent 落地的任务，必须同时满足：

```text
1. 多步骤：一次 LLM 调用无法完成，需要多轮推理或多次工具调用
2. 信息分散：答案需要从多个来源获取和整合
3. 有容错空间：单步失败可以重试或降级，而不是整个流程崩溃
4. 有明确终止条件：能判断任务完成还是未完成
```

不适合 Agent 的任务：

```text
- 单次问答（直接 RAG 或 LLM 即可）
- 实时性要求极高（Agent 多步延迟难以接受）
- 结果不可容错（如金融交易、医疗决策，需要更强的确定性保障）
```

### 8.2 从现有系统找入口

后端工程师的优势：熟悉业务系统。入口往往在你已知的地方。

**方法：扫描"人工重复操作"**

```text
场景扫描问题：
- 有没有人在反复做"查询A → 判断 → 查询B → 汇总"这类事情？
- 有没有需要同时参考多个文档/数据源才能回答的问题？
- 有没有需要人工介入但本质上是规则+知识检索的审批流程？
```

常见高价值入口：

| 场景 | Agent 替代的人工动作 | 核心工具 |
|------|---------------------|---------|
| 客服/支持 | 查知识库 → 判断问题类型 → 生成回答 | RAG tool + 工单 API |
| 代码 Review | 读代码 → 查规范 → 输出意见 | File tool + RAG tool |
| 数据分析报告 | 查数据库 → 计算指标 → 写报告 | SQL tool + 图表 tool |
| 合同/文档审核 | 读文档 → 对照规则 → 标注问题 | RAG tool + 结构化输出 |
| 运维告警分析 | 查日志 → 查监控 → 定位原因 | Log tool + Metrics tool |

### 8.3 落地三步法

```text
Step 1：固化流程（Workflow Agent）
    先把人工流程转成固定步骤的 Workflow Agent
    不要一开始就做动态 Agent，先让流程跑通

Step 2：加入 Human-in-the-loop
    关键决策节点让人审核后再继续
    降低风险，同时收集真实反馈数据

Step 3：用数据驱动自动化比例
    根据 Step 2 积累的数据，识别哪些节点人工几乎总是批准
    逐步把这些节点自动化，减少人工干预
```

### 8.4 内部工具优先原则

生产落地的最低风险路径：

```text
内部工具（仅影响自己团队）
    ↓
内部平台（影响公司内部用户）
    ↓
外部产品（影响真实用户）
```

从内部工具起步的原因：

- 用户容忍度高，出错代价低。
- 需求清晰，容易验证效果。
- 可以快速迭代，不需要严格的发布流程。

典型第一个落地项目：**内部知识库问答 Agent**（你的 RAG 已经做完了，加一层 Agent Loop 即可上线）。

---

## 9. 工程现场

场景：Agent 在生产中频繁"卡住"——工具调用成功但 Agent 不继续推进。

排查后常见是三件事：

1. Tool 返回格式 LLM 无法解析，导致 Agent 判断任务已完成。
2. messages 历史过长触发 context truncation，关键状态被截断。
3. 没有设置最大步数，Agent 在某个循环里反复调用同一工具。

修复顺序：

1. 统一 Tool 返回格式，加 schema 校验。
2. 对 messages history 做滑动窗口或摘要压缩。
3. 强制设置 `max_steps`，超限后返回当前最佳结果。

---

## 10. 文件组织（按顺序阅读）

**阶段一：Agent 核心引擎**

- `00_llm_basics.md`         — System Prompt / Few-shot / CoT / Structured Output / JSON Schema
- `01_tool_calling.md`       — Function Calling 原理与工程实现
- `02_agent_loop.md`         — ReAct 原理与手写 Agent Loop
- `03_memory.md`             — Short-term / Long-term / Working Memory
- `04_streaming.md`          — 流式输出工程实现（SSE / async）

**阶段二：RAG Agent**

- `05_rag_as_tool.md`        — 把 RAG 链路封装为 Agent 工具
- `06_agentic_rag.md`        — Agentic RAG 与 Self-RAG 模式

**阶段三：编排框架**

- `07_langgraph_basic.md`    — StateGraph, Node, Edge 核心概念
- `08_langgraph_advanced.md` — 并发节点, 子图, Human-in-the-loop
- `09_multi_agent.md`        — Supervisor 模式与 Swarm 模式
- `10_mcp.md`                — MCP 协议与工具标准化

**阶段四：生产系统**

- `11_observability.md`      — LangSmith / OpenTelemetry 接入
- `12_evaluation.md`         — Agent 评估体系
- `13_guardrails.md`         — 安全、权限、输出校验
- `14_cost_control.md`       — Token 预算与成本优化
