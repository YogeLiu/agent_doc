# RAGAS：用 4 个指标评估你的 RAG 系统

RAG 系统搭好了，效果"好不好"怎么说清楚？光靠肉眼看几个问答样例远远不够——检索召回的文档是否真的相关？生成的答案有没有"脑补"？这些问题没有一套量化方法就只能靠感觉。

RAGAS 就是解决这个问题的评估框架。它把 RAG 系统拆成检索和生成两段，用 4 个核心指标分别度量各自的质量，基于 LLM-as-judge 自动化运行，不依赖大量人工标注。

---

## 先看全局：四个指标的定位

| 评估环节 | 指标                             | 核心问题                             | 需要的数据                                      |
| -------- | -------------------------------- | ------------------------------------ | ----------------------------------------------- |
| 检索器   | 上下文精度（Context Precision）  | 找回来的，排在前面的有多少是有用的？ | user_input + retrieved_contexts + **reference** |
| 检索器   | 上下文召回率（Context Recall）   | 该找到的关键信息，有没有被漏掉？     | user_input + retrieved_contexts + **reference** |
| 生成器   | 忠实度（Faithfulness）           | 答案有没有脱离检索结果乱编？         | user_input + response + retrieved_contexts      |
| 生成器   | 答案相关性（Response Relevancy） | 答案有没有直接回答问题？             | user_input + response                           |

一句话总结：

- **上下文精度** = 检索的精确率（关注排名质量）
- **上下文召回率** = 检索的召回率（关注信息完整性）
- **忠实度** = 生成的事实一致性（防止幻觉）
- **答案相关性** = 生成的意图对齐度（防止跑题）

---

## RAG 系统的核心问题

RAG 是两端耦合的系统：检索器决定了 LLM 能"看到什么"，LLM 决定了用户最终"得到什么"。问题出在哪一段，要分开看。

常见的失效模式有两类：

**检索问题**：

- 召回的文档不相关，把噪音送给了 LLM
- 关键信息没被召回，LLM 巧妇难为无米之炊

**生成问题**：

- LLM 没有忠实依赖检索结果，自己"发挥"了
- 答案绕远路，没有直接回答用户的问题

RAGAS 的 4 个指标，就是针对这两类问题设计的。

---

## 四个核心指标

### 指标一：上下文精度（Context Precision）

**衡量什么**：检索器召回的文档里，排名靠前的有多少是真正有用的。

**通俗理解**：把检索器比作图书馆管理员。用户问一个问题，管理员搬来一摞书。上下文精度衡量的不只是这摞书里有多少是有帮助的，还要看有帮助的书有没有放在最上面（排名靠前）。

**计算逻辑**：

RAGAS 使用的是**加权累计精度（Weighted Cumulative Precision）**，而非简单的比例，核心思路是"排名越靠前的相关文档贡献越大"：

```
Context Precision@K = Σ(Precision@k × v_k) / 总相关文档数
```

其中：

- `Precision@k` = 前 k 个结果中相关文档数 / k
- `v_k` ∈ {0, 1}，表示第 k 个结果是否相关

举例：检索返回 [相关, 不相关, 相关, 不相关]：

- k=1：Precision@1 = 1/1 = 1.0，v₁=1，贡献 1.0
- k=3：Precision@3 = 2/3 ≈ 0.67，v₃=1，贡献 0.67

这意味着"相关文档排在第 1 位"比"排在第 3 位"得分更高，符合实际使用体验——LLM 更容易利用靠前的上下文。

**需要参考答案（reference）**：RAGAS 用 LLM 判断每个检索 chunk 是否对生成参考答案有帮助，因此必须提供 ground truth。

---

### 指标二：上下文召回率（Context Recall）

**衡量什么**：应该被检索到的关键信息，有没有被漏掉。

**通俗理解**：用户问"年假申请流程和天数规定"，正确答案需要 A 书（讲流程）和 B 书（讲天数）的信息。管理员只找回了 A 书，漏了 B 书，导致答案信息不完整。

**计算逻辑**：

```
Context Recall = 被检索到的 reference 信息点数 / reference 中的总信息点数
```

RAGAS 的实现方式：

1. 从 reference 中提取所有核心信息点（claims）
2. 逐一检查每个 claim 是否能在检索结果里找到依据
3. 能找到依据的 claim 数 / 总 claim 数 = 召回率

**同样需要参考答案（reference）**。

---

### 指标三：忠实度（Faithfulness）

**衡量什么**：生成的答案有没有超出检索结果的范围，有没有"瞎编"。

**通俗理解**：好的 RAG 系统，LLM 应该扮演一个"编辑"而非"作者"——它只负责把检索到的信息整理成通顺的答案，而不是自己创作内容。忠实度衡量的就是 LLM 有没有越界。

**计算逻辑**：

```
Faithfulness = 有上下文支撑的陈述数 / 答案中的总陈述数
```

实现方式用 LLM-as-judge，分两步：

1. **拆解**：把生成的答案拆成若干独立陈述（statements）
2. **核验**：对每个陈述，判断能否从检索结果中推导出来

每个陈述的判断结果只有两类：

- **Supported**：能从检索上下文中找到依据
- **Not Supported**：无法从上下文中找到依据（LLM 在编）

忠实度低的典型症状：LLM 把两份文档的内容混在一起，或者用训练时的知识填补了检索结果的空白。

**不需要参考答案**，只需要 user_input + response + retrieved_contexts。

---

### 指标四：答案相关性（Response Relevancy）

**衡量什么**：答案有没有直接回答用户的问题，而不是答非所问或绕远路。

**通俗理解**：一个好的答案，应该能让人"反推"出问题是什么。如果答案高度切题，从答案出发很容易猜到原始问题；如果答案跑题，从答案几乎猜不出用户在问什么。

**计算逻辑**（RAGAS 的巧妙设计）：

1. 用 LLM 根据生成的答案，反向推测出 N 个可能的问题（默认 N=3）
2. 把推测出的问题和真实的原始问题分别转成向量
3. 计算余弦相似度，取平均值作为得分

```
Response Relevancy = avg(cosine_similarity(推测问题ᵢ, 原始问题))
```

举例对比：

| 原始问题       | 生成答案                        | 推测问题相似度          | 得分  |
| -------------- | ------------------------------- | ----------------------- | ----- |
| 年假怎么申请？ | 需提前 3 天，找经理审批。       | "年假申请流程？" (0.95) | 高 ✅ |
| 年假怎么申请？ | 年假天数按工龄算，1-3 年 5 天。 | "年假有多少天？" (0.45) | 低 ❌ |

**不需要参考答案**，只需要 user_input + response。该指标计算时需要 embedding 模型。

---

## 快速上手 RAGAS（v0.4）

### 安装

```bash
pip install ragas

# 按需安装 LLM provider
pip install langchain-anthropic   # Claude
pip install langchain-openai      # OpenAI
```

### 准备测试数据集

RAGAS v0.2 起字段名有重大变更，v0.4 使用以下字段名：

| v0.1（旧）     | v0.2+（新）          |
| -------------- | -------------------- |
| `question`     | `user_input`         |
| `answer`       | `response`           |
| `contexts`     | `retrieved_contexts` |
| `ground_truth` | `reference`          |

```python
from ragas import EvaluationDataset

samples = [
    {
        "user_input": "年假怎么申请？",
        "response": "年假需提前3天提交申请，经部门经理审批后生效。",
        "retrieved_contexts": [
            "年假申请需提前3天，填写OA系统中的申请表...",
            "部门经理有权批准或拒绝申请..."
        ],
        "reference": "年假需提前3天申请，经部门经理批准。"
    },
    {
        "user_input": "加班工资怎么算？",
        "response": "工作日加班按1.5倍计算，节假日加班按3倍计算。",
        "retrieved_contexts": [
            "加班分为工作日加班和节假日加班，工作日加班按1.5倍..."
        ],
        "reference": "工作日加班1.5倍，休息日2倍，节假日3倍。"
    }
]

dataset = EvaluationDataset.from_list(samples)
```

### 配置评估用的 LLM

**方式一：使用 Claude**

```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = LangchainLLMWrapper(
    ChatAnthropic(model="claude-opus-4-6")
)
# ResponseRelevancy 需要 embedding 模型计算语义相似度
evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
```

**方式二：使用 OpenAI**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
```

**方式三：使用 llm_factory（v0.4 推荐方式，通过 LiteLLM 路由）**

```bash
pip install litellm
```

```python
from ragas.llms import llm_factory

evaluator_llm = llm_factory("claude-opus-4-6")
```

### 运行评估

```python
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
)

results = evaluate(
    dataset=dataset,
    metrics=[
        ContextPrecision(),
        ContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_emb,  # ResponseRelevancy 需要
)

print(results)
# {
#   'context_precision': 0.78,
#   'context_recall': 0.81,
#   'faithfulness': 0.85,
#   'response_relevancy': 0.92
# }

# 转为 DataFrame 方便逐条分析
df = results.to_pandas()
print(df)
```

---

## 自动生成测试数据集

手动构造测试数据集费时费力，RAGAS 提供 `TestsetGenerator`，可以从文档库自动生成问答对：

```python
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader

# 加载知识库文档
loader = DirectoryLoader("./docs", glob="**/*.txt")
docs = loader.load()

# 自动生成测试集
generator = TestsetGenerator(llm=evaluator_llm, embedding_model=evaluator_emb)
testset = generator.generate_with_langchain_docs(docs, testset_size=50)

# 转为 EvaluationDataset
dataset = testset.to_evaluation_dataset()
```

> 注意：自动生成的测试集比真实用户 query 更"规范整洁"，用于快速迭代验证可以，最终验证建议补充真实流量数据。

---

## 分析评估结果

### 分数解读参考

| 指标         | < 0.5                      | 0.5 - 0.7              | > 0.8        |
| ------------ | -------------------------- | ---------------------- | ------------ |
| 上下文精度   | 相关文档排名混乱、噪音严重 | 召回有冗余，需优化排序 | 检索精度较好 |
| 上下文召回率 | 关键信息大量遗漏           | 信息不完整             | 召回较为完整 |
| 忠实度       | LLM 严重在编答案           | 有一定幻觉，需关注     | 较为可靠     |
| 答案相关性   | 大量答非所问               | 部分跑题               | 意图对齐良好 |

### 根据分数定位问题

```
上下文精度低  → 检索排序问题：Reranker 策略需调整、相似度阈值需收紧、chunk 划分太细
上下文召回率低 → 索引覆盖问题：文档未入库、分块切断了关键信息、embedding 模型不适配领域
忠实度低     → LLM 约束问题：system prompt 未限制"只依据上下文回答"、temperature 过高
答案相关性低  → 生成策略问题：prompt 未要求直接回答、答案含不必要的免责声明或套话
```

### 组合分析

指标组合往往比单个指标更有诊断价值：

| 组合情况          | 可能原因                                              |
| ----------------- | ----------------------------------------------------- |
| 精度低 + 忠实度低 | 检索引入大量噪音，LLM 被迫"脑补"                      |
| 召回低 + 忠实度高 | 检索漏掉关键信息，LLM 只用了有限的上下文但很老实      |
| 精度高 + 相关性低 | 检索准确，但 prompt 没有引导 LLM 直接作答             |
| 全部偏低          | 通常是 chunk 策略问题——关键信息被切碎分散在多个 chunk |

---

## 局限性

RAGAS 好用，但要清楚它的边界：

1. **依赖 LLM-as-judge**：评估质量受 judge 模型影响，用不同模型评估可能得到不同分数，同一批次间也存在一定随机性

2. **contexts 与 LLM 实际输入可能不一致**：RAGAS 用检索阶段的 retrieved_contexts 做评估，但实际系统中 Reranker 可能重排结果、上下文可能被压缩截断后再传给 LLM，导致评估对象与实际输入不一致

3. **忠实度 ≠ 事实正确**：忠实度高只代表"答案忠实于检索结果"，如果知识库本身有错误信息，忠实度高反而意味着错误答案被可靠地复现了

4. **Reference 质量决定评估上限**：Context Precision 和 Context Recall 都依赖 reference 的质量。reference 写得不全面，召回率会被低估；reference 有歧义，精度评分会不稳定

5. **不适合在线实时评估**：LLM-as-judge 调用成本较高，适合离线批量评估，不适合对每次请求实时打分

---

## 小结

RAGAS 给 RAG 系统装上了一块仪表盘：

- 检索不准、排名混乱？看**上下文精度**
- 检索不全、关键信息遗漏？看**上下文召回率**
- LLM 在编、答案幻觉？看**忠实度**
- 答非所问、绕远路？看**答案相关性**

四个指标各司其职，覆盖了 RAG 系统最常见的失效模式。搭配定期更新的测试集和 bad case 分析，就能建立起持续有效的 RAG 评估闭环。
