# RAG 评估：如何量化你的检索增强系统

做 RAG 系统的人，十有八九会被一个问题问到："效果怎么样？"

这个问题看似简单，回答起来却不容易。RAG 是一个两端耦合的系统——检索决定了生成的上限，生成又放大了检索的问题。你说"效果不错"，得说清楚是哪个环节不错；说"有问题"，得说清楚是召回不足还是排序混乱、还是生成在编答案。

这一篇聊聊 RAG 评估的整体思路和实操方法。

---

## 评估的层次

RAG 系统的评估可以分成三个层次，每个层次关注不同的东西：

**第一层：检索质量**

检索是根基。检索召回的东西对不对、好不好，直接决定了后面怎么生成。这一层关注的是"找不找得到"和"排得对不对"。

**第二层：生成质量**

检索结果送到 LLM，生成最终回答。这一层关注的是"回答得怎么样"——有没有答非所问、有没有事实错误、信息是否完整。

**第三层：端到端体验**

这是用户视角的评估。不只是答得对不对，还包括回答得快不快、系统稳不稳定、能不能持续产出。

很多团队的问题在于：只盯着第三层（看回答对不对），忽略了第一层（检索到底有没有召回对的东西）。结果调来调去，发现是检索召回的料本身就有问题。

---

## 检索层评估指标

检索评估的核心问题是：给定一个查询，系统返回的 Top-K 结果里，有多少是真正相关的？

### Recall@K

最直观的指标。公式很简单：

```
Recall@K = (检索到的相关文档数) / (数据库中相关文档总数)
```

比如某个查询实际有 10 篇相关文档，系统返回 Top 20 里命中了 7 篇，那么 Recall@20 = 7/10 = 70%。

**这个指标的坑**：分母是"所有相关文档"，但实际项目中你很难知道"所有"到底是多少。通常的做法是用一个标注好的测试集，假设测试集里的标注就是全集。

**工程中的取值**：通常看 Recall@5、Recall@10、Recall@20。如果业务场景是"帮助用户找资料"，Recall@20 是比较合理的关注点；如果场景是"精确问答"，可能更关心 Recall@3 或 Recall@5。

### Precision@K

与 Recall 对应的指标：

```
Precision@K = (检索到的相关文档数) / (K)
```

Precision@K 关心的是"返回的 K 个结果里，有多少是相关的"。这个指标在 RAG 场景下通常不是首要关注点，因为我们可以靠增大 K 来提高召回——代价是后面送给 LLM 的上下文变长、成本变高。

**一个实用的观察**：在 RAG 里，Precision@K 和 Recall@K 经常是矛盾的。增大 K 会提升 Recall，但会降低 Precision；减小 K（只召回很少的结果） Precision 很高，但可能漏掉相关文档。调参的本质是在这个矛盾里找平衡。

### MRR (Mean Reciprocal Rank)

MRR 关注"第一个相关结果排在哪里"：

```
MRR = (1/N) * Σ (1/rank_i)
```

其中 rank_i 是第 i 个查询的第一个相关文档的排名。

举例：
- 查询1：相关文档排在第1位 → 1/1 = 1
- 查询2：相关文档排在第3位 → 1/3 ≈ 0.33
- 查询3：没有相关文档 → 0

MRR = (1 + 0.33 + 0) / 3 ≈ 0.44

**适用场景**：MRR 特别适合"搜到一个就够用"的场景。比如搜公司名、搜产品型号，用户点第一个结果就满意了，不需要看后面一排。

### NDCG (Normalized Discounted Cumulative Gain)

这是最"全面"的检索指标，考虑了"相关程度"不只是"相关/不相关"两档：

```
NDCG@K = DCG@K / IDCG@K
```

- DCG@K = Σ (rel_i / log2(i+1))，i 是排名，rel_i 是第 i 个结果的相关性分数
- IDCG@K 是"理想情况下的 DCG"——把所有相关文档按相关性排序后的 DCG

相关性分数可以这样设计：
- 非常相关：3 分
- 相关：2 分
- 有点相关：1 分
- 不相关：0 分

**为什么 NDCG 有用**：它同时惩罚了两件事——"相关文档排得太靠后"和"不相关文档排得太靠前"。如果你的排序算法把相关文档从第 5 位挪到第 1 位，NDCG 会明显上升；如果把不相关文档从第 20 位挪到第 2 位，NDCG 会明显下降。

**实际使用建议**：NDCG 需要标注每个文档的相关等级，工作量比二元标注（相关/不相关）大很多。如果团队资源有限，先用 Recall@K + MRR 足够了；资源充裕再上 NDCG。

### Hit Rate

Hit Rate 是最简单的指标：查询一次，有没有命中相关文档？

```
Hit Rate@K = (至少命中 1 个相关文档的查询数) / (总查询数)
```

这个指标和 MRR 配合使用效果很好。Hit Rate 告诉你"能不能找到"，MRR 告诉你"找得有多快"。

---

## 生成层评估指标

检索做对了，生成不一定对。检索召回 10 篇相关文档，LLM 可能只看了前 3 篇，也可能看了全部但理解错了，还可能自己编了一个测试集里根本没有的答案。

### Faithfulness (忠实度)

这个指标评估的是"LLM 说的答案能不能从检索到的上下文里推出来"。

举一个典型的失败案例：
- 检索到的上下文：["苹果公司在2024年发布了iPhone 16", "华为Mate 70主打影像功能"]
- LLM 生成："苹果在2024年发布了Mate 70"——这就是不忠实的答案，混淆了两条信息。

**实现方式**：通常用 LLM-as-judge 来做。给 LLM 两样东西——检索到的上下文和问题，让它判断答案里的每个陈述是否能在上下文里找到依据。

```python
# RAGAS 的 faithfulness 评估 prompt 大致逻辑
prompt = f"""
Given the context and question, evaluate whether each statement 
in the answer can be derived from the context.

Context: {retrieved_context}
Question: {question}
Answer: {generated_answer}

For each statement in the answer, output:
- "SUPPORTED": the statement can be derived from context
- "NOT_SUPPORTED": the statement cannot be derived from context  
- "IRRELEVANT": the statement is irrelevant to the question
"""
```

Faithfulness 得分 = (Supported 的陈述数) / (总陈述数)

### Answer Relevance (答案相关性)

这个指标评估的是"答案有没有回答问题"。

常见的问题：
- 答案"太长"——绕了一圈没说到点子上
- 答案"太短"——只给了部分信息
- 答案"跑题"——回答的不是用户问的那个问题

**实现方式**：同样用 LLM-as-judge，让它从"是否直接回答了问题"、"信息是否完整"、"有没有多余信息"几个维度打分。

### Context Precision & Context Recall

这两个指标把检索和生成连接起来看：

**Context Precision**：检索到的 K 个文档里，有多少是真正有用的？用来检测"检索召回了一堆噪音"的问题。

**Context Recall**：应该检索到的上下文有没有被完整召回？用来检测"关键信息被漏掉"的问题。

Context Recall 的计算需要一些技巧——你得先知道"理想情况下应该召回什么"。一种做法是让 LLM 根据问题生成一个"理想答案"，然后看这个理想答案里的每个事实性陈述在检索到的上下文里出现了没有。

---

## 自动化评估工具

手写评估脚本太累了，以下工具可以帮你：

### RAGAS

目前 RAG 评估的事实标准。直接装、给定测试集就能跑：

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(
    dataset=rag_test_set,  # 包含 question, answer, contexts, ground_truth 的数据集
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

RAGAS 的核心逻辑就是上面说的 faithfulness 和 answer relevance，用 GPT-4 或 Claude 作为 judge 模型来打分。

**局限性**：它假设"检索到的 contexts"就是"LLM 实际看到的 contexts"。如果你的系统做了 context 压缩或截断，这个假设就不成立了。

### TruLens

比 RAGAS 更灵活的地方在于它支持"自定义评估函数"：

```python
from trulens_eval import Feedback, TruLlama

# 定义 faithfulness 反馈
faithfulness = Feedback(
    TruLlama().groundedness,
    name="Faithfulness"
).on_input_output()

# 评估
tru = TruLlama(app_id="my_rag_app", feedbacks=[faithfulness])
tru.evaluate(data)
```

TruLens 另外一个亮点是支持"过程追踪"——能看到每个检索结果对最终答案的贡献度，方便定位问题。

### 自建评估系统

如果你的团队有余力，自建评估系统有几个方向：

**1. 固定测试集 + 人工标注**

这是最可靠的方式。找业务专家标注 200-500 个有代表性的查询，覆盖常见场景、边界case、bad case。每周或每次大版本更新后跑一遍，看指标趋势。

**2. A/B 测试**

线上流量分 A/B 两组，一组用新策略，一组用旧策略，观察：
- 用户点击率
- 用户平均对话轮次
- 人工抽检回答质量
- 任务完成率

A/B 测试是验证"改动的实际价值"的终极方式——指标提升不等于用户体验提升。

**3. LLM-as-judge 自动化**

用更强的 LLM（比如 GPT-4）来评估较弱的 LLM 的输出。这个方法有争议——"用更聪明的模型评估不那么聪明的模型"这个逻辑本身就有问题。但作为快速迭代的辅助手段，它足够好用。

---

## 检索层：怎么判断召回好不好

检索是 RAG 的第一环，也是最容易量化的一环。以下是工程实践中的做法：

### 建立测试集

测试集的质量决定了你评估的上限。建议分三类：

1. **高频 query**：业务里出现最多的 100-200 个查询，每个 query 标注 3-5 个相关文档
2. **边界 case**：容易召回失败的 query——短 query、长 query、含专有名词的 query、多意图 query
3. **Bad case 库**：线上用户反馈"回答不对"的case，定期补充到测试集

### 看哪些指标

**日常监控**：Recall@10 + MRR + Hit Rate@10。这三个指标综合反映了召回的"全不全"和"准不准"。

**深度分析**：按 query 类型分维度看。比如：
- 短 query（<5 tokens）：Recall@10 是多少？
- 含数字的 query：Recall@10 是多少？
- 多意图 query：Hit Rate 是多少？

这样能定位到底是"整体不好"还是"某类 query 不好"。

### 一个常见的调试思路

假设 Recall@10 从 80% 跌到了 70%，怎么排查？

第一步：按 query 长度、query 类型、召回来源（dense / sparse / rerank）分维度看，确认是全面下跌还是某类 query 变差。

第二步：如果某类 query 变差，看是dense召回不行还是 sparse 召回不行。可以用消融实验——关掉某一路召回，看 Recall@10 掉多少。

第三步：如果是 dense 召回的问题，可能是 embedding 模型该换了；如果是 sparse 召回的问题，可能是分词策略或 chunking 策略需要调整。

---

## 生成层：怎么判断回答好不好

生成的问题更难量化，因为"回答得好不好"本身带有主观色彩。

### 分层评估策略

**第一层：自动化指标**

用 RAGAS 跑 faithfulness + answer relevance。这两个指标能帮你快速发现"明显的问题"——比如 LLM 明显在编答案（faithfulness 低）、或者答非所问（answer relevance 低）。

**第二层：人工抽检**

自动化指标只能筛掉"明显差的case"，对于"中等质量"的case 还是得靠人看。

建议的抽检节奏：
- 每日：随机抽 20 个 case 人工过一遍
- 每周：按 query 类型各抽 5-10 个case，做分类统计
- 每次大改动后：全面过一遍测试集

**第三层：业务指标**

最终还是要看业务效果：
- 用户是否点击了"重新生成"？
- 用户是否在对话中明确表达了不满？
- 任务完成率（如果是任务型 RAG）有没有变化？

---

## 端到端：怎么判断系统稳不稳

这一块经常被忽略，直到线上出了事故。

### 系统指标

- **延迟**：P50 / P95 / P99 查询延迟。RAG 系统延迟通常在 200ms-2s 之间，取决于检索路数、rerank 模型、LLM 响应时间。
- **可用性**：服务是否挂了、错误率是多少。
- **空结果率**：有多少查询什么都没召回？空结果率超过 5% 就需要关注了。

### 成本指标

- **LLM 调用成本**：每次 query 花了多少 LLM token
- **检索成本**：向量数据库的 QPS 峰值、存储量

### 监控看板建议

建议搭一个实时看板，至少包含：
- 过去 1 小时的 Query Volume、延迟 P95、空结果率
- 过去 24 小时的 Recall@10、Faithfulness、Answer Relevance 趋势
- 过去 1 周的 bad case 分布

---

## 常见问题与回答

**Q：评估指标这么多，应该重点看哪个？**

A：分阶段。启动阶段看 Recall@10 + MRR——先保证能召回对的东西；迭代阶段看 Faithfulness + Answer Relevance——确保 LLM 没在瞎编；上线后看端到端指标——延迟、可用性、成本。

**Q：测试集要多大才够用？**

A：100-200 个 query 是起步，500 个以上才有统计意义。关键是覆盖度——高频 case、边界 case、bad case 都要有。

**Q：可以用 LLM 生成测试 query 吗？**

A：不建议。LLM 生成的 query 往往比真实用户 query 更规范、更"整洁"，无法反映真实场景的复杂性。真实 query 里的拼写错误、碎片化表达、歧义性才是 RAG 系统的真正考验。

**Q：Recall 很高但用户还是反馈"找不到答案"，为什么？**

A：可能的原因：
1. Recall 高但 Precision 低——召回的东西虽然相关但不是最相关的，前面的被淹了
2. 检索到的信息太碎片化，LLM 无法整合出完整答案
3. Chunking 策略有问题，关键信息被切到了不同 chunk 里

**Q：线上和离线的指标趋势不一致，怎么回事？**

A：通常是测试集和线上流量分布不一致。检查：测试集的 query 分布（长度、类型、领域）是否和线上真实流量一致？测试集的文档覆盖率是否和线上知识库一致？

---

## 小结

RAG 评估不是一件事，而是一套组合拳：

1. **检索层**：用 Recall@10 + MRR + Hit Rate 监控召回质量，用测试集消缺定位具体问题
2. **生成层**：用 Faithfulness + Answer Relevance 做自动化筛选，用人工抽检验证
3. **端到端**：监控延迟、可用性、成本等系统指标

最重要的是建立**可持续的评估闭环**——测试集定期更新、指标定期跑、bad case 定期分析。没有这个闭环，调参就是盲人摸象。

---
