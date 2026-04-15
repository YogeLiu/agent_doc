# Evaluation：Agent 的评估体系

Agent 和普通 LLM 应用最大的评估难点：**多步任务的正确性不等于每步正确性的简单加和。**

一个 Agent 可能：
- 每步看起来都合理，但最终结果错误（路径正确 × 结论错误）
- 走了弯路但最终结果正确（路径低效 × 结论正确）
- 在某一步用错了工具但后续自我修正了（中间错误 × 最终正确）

这三种情况在评估时需要区分对待。

---

## 评估的三个层次

```text
Layer 1：最终结果评估（Task-level）
    Agent 完成任务了吗？答案正确吗？

Layer 2：路径评估（Trajectory-level）
    Agent 的执行路径合理吗？步数效率如何？

Layer 3：组件评估（Component-level）
    检索准确吗？工具调用正确吗？LLM 推理合理吗？
```

---

## Layer 1：最终结果评估

### 任务成功率

最直接的指标：Agent 完成任务了吗？

```python
from dataclasses import dataclass

@dataclass
class EvalCase:
    input: str                    # 用户输入
    expected_output: str          # 期望答案（可以是关键词/要点）
    expected_tools: list[str]     # 期望调用的工具列表
    max_acceptable_steps: int     # 可接受的最大步数
    tags: list[str] = None       # 分类标签（如 "simple", "multi-hop"）


async def eval_task_success(agent_fn, cases: list[EvalCase]) -> dict:
    results = []
    for case in cases:
        trace = await agent_fn(case.input)

        # 判断答案是否正确
        is_correct = await judge_answer_correctness(
            question=case.input,
            expected=case.expected_output,
            actual=trace.final_output
        )

        results.append({
            "input": case.input,
            "correct": is_correct,
            "steps": trace.step_count,
            "tokens": trace.total_tokens,
            "latency_ms": trace.duration_ms,
            "tools_used": [s.name for s in trace.spans if s.span_type == "tool_call"]
        })

    success_rate = sum(1 for r in results if r["correct"]) / len(results)
    return {"success_rate": success_rate, "details": results}


async def judge_answer_correctness(question: str, expected: str, actual: str) -> bool:
    """用 LLM 判断答案是否正确（LLM-as-Judge）"""
    prompt = f"""判断"实际答案"是否正确回答了问题。参考"期望答案"的关键要点。

问题：{question}
期望答案要点：{expected}
实际答案：{actual}

判断标准：
- 如果实际答案覆盖了期望答案的核心要点，判定为正确
- 如果实际答案遗漏了关键信息或包含明显错误，判定为错误
- 表述不同但意思正确算正确

回答（只回复 CORRECT 或 INCORRECT）："""
    result = await call_llm(prompt)
    return result.strip().upper() == "CORRECT"
```

### Faithfulness（忠实度）

答案是否忠实于检索到的内容，有没有凭空编造：

```python
async def eval_faithfulness(question: str, context: str, answer: str) -> float:
    """评估答案对检索结果的忠实度（0-1）"""
    prompt = f"""分析"答案"中的每个声明，判断是否有"上下文"支撑。

上下文：
{context[:2000]}

答案：
{answer}

步骤：
1. 列出答案中的所有事实声明（每行一个）
2. 对每个声明，标注 SUPPORTED（有上下文支撑）或 UNSUPPORTED

输出格式：
声明1 | SUPPORTED
声明2 | UNSUPPORTED
..."""
    result = await call_llm(prompt)

    lines = [l.strip() for l in result.strip().split("\n") if "|" in l]
    if not lines:
        return 1.0
    supported = sum(1 for l in lines if "SUPPORTED" in l.upper())
    return supported / len(lines)
```

---

## Layer 2：路径评估

### 步数效率

```python
def eval_step_efficiency(traces: list[Trace], cases: list[EvalCase]) -> dict:
    """评估 Agent 的步数效率"""
    results = []
    for trace, case in zip(traces, cases):
        actual_steps = trace.step_count
        max_steps = case.max_acceptable_steps

        efficiency = min(1.0, max_steps / actual_steps) if actual_steps > 0 else 0
        results.append({
            "input": case.input,
            "actual_steps": actual_steps,
            "max_acceptable": max_steps,
            "efficiency": efficiency,
            "over_budget": actual_steps > max_steps
        })

    avg_efficiency = sum(r["efficiency"] for r in results) / len(results)
    over_budget_rate = sum(1 for r in results if r["over_budget"]) / len(results)

    return {
        "avg_efficiency": avg_efficiency,
        "over_budget_rate": over_budget_rate,
        "details": results
    }
```

### 工具调用准确率

```python
def eval_tool_accuracy(traces: list[Trace], cases: list[EvalCase]) -> dict:
    """评估 Agent 是否调用了正确的工具"""
    results = []
    for trace, case in zip(traces, cases):
        expected = set(case.expected_tools)
        actual = set(s.name for s in trace.spans if s.span_type == "tool_call")

        precision = len(expected & actual) / len(actual) if actual else 1.0
        recall = len(expected & actual) / len(expected) if expected else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "expected": list(expected),
            "actual": list(actual),
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    return {"avg_tool_f1": avg_f1, "details": results}
```

### 循环检测

```python
def eval_loop_rate(traces: list[Trace]) -> float:
    """检测多少比例的 Trace 存在循环"""
    loop_count = 0
    for trace in traces:
        tool_calls = [
            f"{s.name}:{json.dumps(s.input_data)}"
            for s in trace.spans if s.span_type == "tool_call"
        ]
        from collections import Counter
        counts = Counter(tool_calls)
        if any(c > 2 for c in counts.values()):
            loop_count += 1

    return loop_count / len(traces)
```

---

## Layer 3：组件评估

### 检索评估（复用 RAG 指标）

你已有的 RAG 评估体系直接复用：

```python
# Recall@k：top-k 结果里包含正确答案的比例
# MRR：第一个正确结果的排名倒数均值
# nDCG：考虑排序质量的综合指标

# 用 Ragas 评估
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

ragas_result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

### 检索决策评估

Agent 该检索时检索了吗？不该检索时没检索吗？

```python
@dataclass
class RetrievalDecisionCase:
    input: str
    should_retrieve: bool  # True=需要检索, False=不需要

def eval_retrieval_decision(traces: list[Trace], cases: list[RetrievalDecisionCase]) -> dict:
    correct = 0
    for trace, case in zip(traces, cases):
        actually_retrieved = any(
            s.span_type == "tool_call" and "search" in s.name
            for s in trace.spans
        )
        if actually_retrieved == case.should_retrieve:
            correct += 1

    return {"retrieval_decision_accuracy": correct / len(cases)}
```

---

## 构建评估数据集

评估数据集是评估体系的基础，比评估方法更重要。

### 数据来源

```text
来源1：人工标注（质量最高，成本最高）
    从生产流量里抽样，人工标注期望答案和期望工具

来源2：LLM 生成（质量中等，成本低）
    用 LLM 基于文档库自动生成 QA 对

来源3：生产日志（来源最广，需要清洗）
    从 Trace 里提取"人工标记为满意"的案例
```

### 最小可行数据集

```python
eval_cases = [
    # 简单知识问答
    EvalCase(
        input="rerank 是什么？",
        expected_output="rerank 是对检索召回的候选文档进行重排序，使用 Cross-encoder 提升排序精度",
        expected_tools=["search_knowledge_base"],
        max_acceptable_steps=3,
        tags=["simple", "rag"]
    ),

    # 不需要检索的问题
    EvalCase(
        input="你好",
        expected_output="你好，有什么可以帮助你的吗",
        expected_tools=[],
        max_acceptable_steps=1,
        tags=["simple", "no_retrieval"]
    ),

    # 多跳问题
    EvalCase(
        input="我们的 RAG 系统用的 rerank 模型和业界最佳实践一致吗？",
        expected_output="需要对比我们的 rerank 模型和推荐的 bge-reranker-large",
        expected_tools=["search_knowledge_base"],
        max_acceptable_steps=6,
        tags=["complex", "multi-hop"]
    ),

    # 工具错误恢复
    EvalCase(
        input="查一下 2025 年 Q1 的销售数据",
        expected_output="如果数据库查询失败，应该告知用户无法获取，而不是编造数据",
        expected_tools=["query_database"],
        max_acceptable_steps=4,
        tags=["error_handling"]
    ),
]
```

建议至少准备 50-100 条评估案例，覆盖主要场景和边界情况。

---

## 自动化评估流程

```python
async def run_full_evaluation(agent_fn, cases: list[EvalCase]) -> dict:
    """运行完整评估套件"""
    # 运行所有案例
    traces = []
    for case in cases:
        trace = await agent_fn(case.input)
        traces.append(trace)

    # 多维度评估
    task_eval = await eval_task_success(agent_fn, cases)
    step_eval = eval_step_efficiency(traces, cases)
    tool_eval = eval_tool_accuracy(traces, cases)
    loop_eval = eval_loop_rate(traces)

    # 按 tag 分组统计
    tag_stats = {}
    for trace, case in zip(traces, cases):
        for tag in (case.tags or []):
            if tag not in tag_stats:
                tag_stats[tag] = {"total": 0, "success": 0}
            tag_stats[tag]["total"] += 1
            # ...

    report = {
        "overall": {
            "task_success_rate": task_eval["success_rate"],
            "avg_step_efficiency": step_eval["avg_efficiency"],
            "avg_tool_f1": tool_eval["avg_tool_f1"],
            "loop_rate": loop_eval,
        },
        "by_tag": tag_stats,
        "cost": {
            "total_tokens": sum(t.total_tokens for t in traces),
            "avg_tokens_per_request": sum(t.total_tokens for t in traces) / len(traces),
        }
    }

    return report
```

### 回归测试

每次修改 Agent 后，跑一遍评估，确保没有退化：

```python
def compare_reports(current: dict, baseline: dict, threshold: float = 0.05) -> list[str]:
    """对比当前报告和基线报告，输出退化项"""
    regressions = []
    for metric in ["task_success_rate", "avg_step_efficiency", "avg_tool_f1"]:
        current_val = current["overall"][metric]
        baseline_val = baseline["overall"][metric]
        if current_val < baseline_val - threshold:
            regressions.append(
                f"{metric}: {baseline_val:.3f} → {current_val:.3f} (下降 {baseline_val - current_val:.3f})"
            )
    return regressions
```

---

## LLM-as-Judge 的注意事项

用 LLM 评估 LLM 输出是常见做法，但有几个坑：

```text
问题1：评估 LLM 和被评估 LLM 用同一个模型
    → 可能有系统性偏好，建议用不同模型评估

问题2：评估 prompt 措辞影响判断
    → 多写几个评估 prompt 做交叉验证

问题3：LLM-as-Judge 自身有幻觉
    → 定期人工抽检评估结果的准确性

问题4：一致性
    → 同一个案例多次评估，结果可能不一致
    → 对关键指标取多次评估的平均值
```

---

## 工程现场

场景：评估结果显示任务成功率 85%，上线后用户反馈"感觉比之前差了"。

原因：评估数据集偏简单，没有覆盖用户实际使用的复杂场景。

修复：

1. 从生产 Trace 里抽取"用户标记为不满意"的案例，加入评估集。
2. 按用户场景分布加权评估，而不是等权平均。
3. 加 A/B 测试：新版本先给 10% 流量，对比满意度指标。

---

## 小结

Agent 评估的核心框架：

```text
评估什么：
    Task-level   → 任务成功率、Faithfulness
    Trajectory   → 步数效率、工具准确率、循环率
    Component    → 检索 Recall、检索决策准确率

用什么评估：
    LLM-as-Judge → 快速、可规模化，但有系统偏差
    人工评估     → 最准确，但慢且贵
    自动指标     → 步数、Token、延迟等可自动计算

多久评估一次：
    每次改动  → 跑回归测试
    每周      → 分析生产 Trace 质量
    每月      → 更新评估数据集
```

下一篇讲 Guardrails：如何防止 Agent 做出不安全的行为。
