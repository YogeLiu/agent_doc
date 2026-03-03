# 文本分块（Chunking）

## 概述

文本分块（Chunking）是 RAG（检索增强生成）系统中的关键预处理步骤，指将大型文档分割成更小、更易管理的片段（chunk），以便后续进行向量化和检索。

分块策略的核心矛盾：

- **块太小**：上下文信息不足，检索到的内容缺乏完整语义，LLM 无法生成准确回答
- **块太大**：包含过多无关信息，稀释相关内容的语义权重，检索精度下降；同时可能产生"Lost in the Middle"问题

**分块质量直接影响整个 RAG 系统的性能**，因此需要根据数据类型、查询特点和下游任务选择合适的分块策略。

---

## 一、为什么需要分块？

### 1.1 技术限制

**Embedding 模型的 token 上限：**

大多数 embedding 模型都有最大输入长度限制：

| 模型                         | 最大 token 数 |
| ---------------------------- | ------------- |
| text-embedding-ada-002       | 8,191         |
| text-embedding-3-small/large | 8,191         |
| bge-large-zh                 | 512           |
| bge-m3                       | 8,192         |
| E5-large                     | 512           |

超过上限的文本会被截断，导致信息丢失。

**LLM 上下文窗口限制：**

即使 LLM 支持大上下文窗口（如 128K tokens），把所有检索结果都塞入也会导致：

- 推理成本急剧增加
- "Lost in the Middle" 问题：模型对中间位置内容的关注度显著下降
- 注意力被稀释，响应质量下降

### 1.2 检索精度考量

将文档切分为语义内聚的小块，使得：

- 每个 chunk 聚焦于单一主题或概念
- 向量检索时相关块的语义相似度更高
- 检索结果更精准，减少噪音

### 1.3 向量表示质量

短文本通常比长文本产生更高质量的 embedding，因为：

- 长文本中多个主题混合，导致 embedding 向量"模糊"（语义被平均化）
- 短文本语义集中，embedding 更能捕捉核心含义，检索时相似度计算更精准

### 1.4 分块的"甜蜜点"

好的分块需要同时满足两个目标：

- **对检索友好**：语义聚焦，embedding 质量高，易于向量匹配
- **对生成友好**：上下文完整，LLM 能基于 chunk 独立回答问题

> 判断标准：**一个 chunk 如果人类读了也能理解其含义，LLM 也能理解。** 如果需要前后文才能理解，说明 chunk 太小或边界切得不好。

---

## 二、分块时机：Pre-Chunking vs Post-Chunking

在构建 RAG 系统时，不仅要选择"怎么分块"，还要考虑"何时分块"。

### 2.1 预分块（Pre-Chunking）— 主流方式

**流程：**

```
文档 → 分块 → 向量化 → 存入向量数据库
                               ↓
用户查询 → 向量检索 → 返回相关 chunk → LLM 生成答案
```

**特点：**

- 所有 chunk 在索引阶段已预先计算，查询时直接检索，**延迟低**
- 需要在索引阶段就确定分块策略和参数
- 是目前绝大多数 RAG 系统采用的方式

### 2.2 后分块（Post-Chunking）— 动态方式

**流程：**

```
文档 → 向量化（整文档）→ 存入向量数据库
                               ↓
用户查询 → 检索相关文档 → 对检索到的文档实时分块 → LLM 生成答案
```

**特点：**

- 只对实际被检索到的文档进行分块，**避免对冷数据分块的浪费**
- 支持基于具体查询动态调整分块策略
- 首次访问有额外延迟；已访问文档的分块结果可缓存加速

**适用场景：**

- 文档库极大但查询集中在小部分文档的场景
- 需要根据不同查询动态调整分块粒度的高级系统

---

## 三、分块方法分类

所有分块方法按复杂度和实现方式分为三大类：

```
┌─────────────────────────────────────────────────────────────┐
│  第一类：基础分块（规则驱动，无需 ML/LLM）                       │
│  Fixed-Size / Recursive / Document-Based / Code / Sliding   │
├─────────────────────────────────────────────────────────────┤
│  第二类：智能分块（模型驱动，使用 Embedding 或 LLM）              │
│  Semantic / LLM-Based / Agentic / Adaptive                  │
├─────────────────────────────────────────────────────────────┤
│  第三类：结构增强分块（改变分块粒度或补充上下文）                  │
│  Hierarchical / Parent-Child / Contextual / Proposition /   │
│  Late Chunking                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、第一类：基础分块方法

### 4.1 固定大小分块（Fixed-Size Chunking）

**原理：**

按照固定的字符数或 token 数将文本机械地分割，是最简单的分块方式。

```
文本: "这是第一句话。这是第二句话。这是第三句话。"
chunk_size=10, overlap=2

Chunk 1: "这是第一句话。这是"
Chunk 2: "。这是第二句话。这"
Chunk 3: "。这是第三句话。"
```

**代码示例：**

```python
from typing import List
import re

def word_splitter(source_text: str) -> List[str]:
    source_text = re.sub(r"\s+", " ", source_text)
    return re.split(r"\s", source_text)

def fixed_size_chunking(text: str, chunk_size: int, overlap_fraction: float = 0.2) -> List[str]:
    words = word_splitter(text)
    overlap = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[max(i - overlap, 0): i + chunk_size]
        chunks.append(" ".join(chunk_words))
    return chunks
```

**优缺点：**

| 维度 | 说明                                    |
| ---- | --------------------------------------- |
| 优点 | 实现极简，速度最快，无需任何模型        |
| 缺点 | 可能在句子/词语中间切断，破坏语义完整性 |
| 质量 | ★★☆                                     |
| 速度 | ★★★                                     |
| 成本 | 极低                                    |

**适用场景：** 快速原型验证；结构简单的短文档；对速度要求极高、对质量要求不高的场景

**参数建议：** chunk_size：256～512 tokens；overlap：10%～20%

---

### 4.2 递归字符分块（Recursive Character Chunking）⭐ 通用首选

**原理：**

使用一组有序的分隔符列表，按优先级递归地分割文本。当按高优先级分隔符分割后，若某块仍超过目标大小，则对该块使用下一个分隔符继续分割。

**默认分隔符优先级（以 LangChain 为例）：**

```
["\n\n", "\n", "。", "！", "？", ".", " ", ""]
```

**分割过程示意：**

```
原始文本（太长）
    ↓ 尝试用 "\n\n"（段落）分割
段落 A（合适） | 段落 B（仍太长）
                    ↓ 尝试用 "\n"（行）分割
               行 B1（合适） | 行 B2（仍太长）
                                  ↓ 尝试用 "。"（句子）分割
                             句子 B2a | 句子 B2b（合适）
```

**代码示例（LangChain）：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = splitter.split_text(text)
```

**手动实现：**

```python
from typing import List

def recursive_chunking(text: str, max_chunk_size: int = 1000) -> List[str]:
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []

    separators = ["\n\n", "\n", "。", " "]

    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks, current_chunk = [], ""

            for part in parts:
                test_chunk = current_chunk + separator + part if current_chunk else part
                if len(test_chunk) <= max_chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part

            if current_chunk:
                chunks.append(current_chunk.strip())

            # 对仍过长的块递归处理
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size:
                    final_chunks.extend(recursive_chunking(chunk, max_chunk_size))
                else:
                    final_chunks.append(chunk)
            return [c for c in final_chunks if c]

    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
```

**优缺点：**

| 维度 | 说明                                                         |
| ---- | ------------------------------------------------------------ |
| 优点 | 尽可能在自然边界处分割，语义完整性好；适用性广；框架原生支持 |
| 缺点 | 仍是基于规则，无法理解语义；对没有明显结构的文本效果有限     |
| 质量 | ★★★                                                          |
| 速度 | ★★★                                                          |
| 成本 | 极低                                                         |

**适用场景：** 大多数通用文档（文章、报告、书籍等），**是默认首选策略**

---

### 4.3 文档结构分块（Document-Based Chunking）

**原理：**

利用文档本身的结构化标记（Markdown 标题、HTML 标签、PDF 章节等）作为天然分割边界，按文档逻辑结构分块，结构信息同时存入 metadata。

**Markdown 文档：**

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
# 每个 chunk 携带 metadata: {"H1": "第一章", "H2": "第一节"}
```

**HTML 文档：**

```python
from langchain.text_splitter import HTMLHeaderTextSplitter

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_content)
```

**手动实现（Markdown）：**

```python
from typing import List
import re

def markdown_document_chunking(text: str) -> List[str]:
    header_pattern = r'^#{1,6}\s+.+$'
    lines = text.split('\n')
    chunks, current_chunk = [], []

    for line in lines:
        if re.match(header_pattern, line, re.MULTILINE):
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            current_chunk = [line]
        else:
            current_chunk.append(line)

    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks
```

**优缺点：**

| 维度 | 说明                                                             |
| ---- | ---------------------------------------------------------------- |
| 优点 | 分块边界与文档逻辑完全对齐；结构信息可存为 metadata 用于过滤检索 |
| 缺点 | 强依赖文档格式，格式不规范则效果差；不同格式需不同解析器         |
| 质量 | ★★★★                                                             |
| 速度 | ★★★                                                              |
| 成本 | 低                                                               |

**适用场景：** 格式规范的技术文档（Markdown、HTML）；有明确章节结构的报告

---

### 4.4 代码分块（Code Chunking）

**原理：**

专门针对源代码的分块策略，按代码的逻辑结构（函数、类、方法）而非字符数进行切分，保持代码语法和逻辑完整性。

**代码示例（LangChain）：**

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# 支持 Python、JavaScript、Go、Java、C++ 等多种语言
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100
)
chunks = python_splitter.split_text(python_code)
```

**不同语言的分隔符：**

| 语言       | 主要分隔符                            |
| ---------- | ------------------------------------- |
| Python     | `\nclass `, `\ndef `, `\n\n`          |
| JavaScript | `\nfunction `, `\nconst `, `\nclass ` |
| Go         | `\nfunc `, `\ntype `, `\nvar `        |
| Java       | `\nclass `, `\npublic `, `\nprivate ` |

**优缺点：**

| 维度 | 说明                                           |
| ---- | ---------------------------------------------- |
| 优点 | 保持代码语法完整；函数/类级别的 chunk 语义内聚 |
| 缺点 | 超长函数仍需进一步拆分；不适合非代码文本       |
| 质量 | ★★★★                                           |
| 速度 | ★★★                                            |
| 成本 | 低                                             |

**适用场景：** 代码库、API 文档、Jupyter Notebook

---

### 4.5 滑动窗口分块（Sliding Window Chunking）

**原理：**

固定大小分块的增强版，通过增加 overlap（重叠）确保相邻块之间有内容交叉，避免在边界处切断关键信息。

```
chunk_size=100, overlap=20

Chunk 1: tokens [0, 100)
Chunk 2: tokens [80, 180)   ← 与 Chunk 1 重叠 20 个 token
Chunk 3: tokens [160, 260)
...
```

**为什么需要重叠：**

```
没有重叠时（关键信息落在边界）：
  Chunk 1: "...这种方法的核心思想是利用"
  Chunk 2: "贝叶斯推断来解决..."
  → 用户问"这种方法是什么"时，两块单独都无法回答

有重叠时：
  Chunk 1: "...这种方法的核心思想是利用贝叶斯推断"  ✓ 包含完整信息
  Chunk 2: "利用贝叶斯推断来解决..."
```

**重叠比例建议：**

| 场景             | 建议 overlap |
| ---------------- | ------------ |
| 一般场景         | 10%～20%     |
| 长句子、复杂推理 | 20%～30%     |
| 简单事实型文档   | 5%～10%      |

**注意：** overlap 过大会导致大量冗余存储和检索干扰，overlap 过小则无法有效防止边界切断问题。

---

## 五、第二类：智能分块方法

### 5.1 语义分块（Semantic Chunking）

**原理：**

利用 embedding 模型计算相邻句子之间的语义相似度，在语义发生显著跳变的位置切分，确保每个 chunk 内部语义高度相关。

**实现步骤：**

```
1. 将文本按句子分割（句号、换行等）
2. 对每个句子生成 embedding 向量
3. 计算相邻句子对的余弦相似度
4. 找到相似度骤降的位置（"语义断点"）
5. 在断点处切分文本，合并相似度高的相邻句子
```

**语义断点检测方法：**

| 方法                   | 说明                                        | 参数                             |
| ---------------------- | ------------------------------------------- | -------------------------------- |
| 百分位法（Percentile） | 相似度低于第 X 百分位数的位置为断点         | `breakpoint_threshold_amount=95` |
| 标准差法（Std Dev）    | 相似度低于 `均值 - X * 标准差` 的位置为断点 | `breakpoint_threshold_amount=3`  |
| 四分位法（IQR）        | 用 IQR 检测异常低的相似度位置               | —                                |

**代码示例（LangChain）：**

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)
chunks = chunker.split_text(text)
```

**优缺点：**

| 维度 | 说明                                                                        |
| ---- | --------------------------------------------------------------------------- |
| 优点 | chunk 内部主题高度连贯；块大小灵活，不强制固定长度                          |
| 缺点 | 需对所有句子做 embedding，**预处理成本高**；速度慢；效果依赖 embedding 质量 |
| 质量 | ★★★★                                                                        |
| 速度 | ★★☆                                                                         |
| 成本 | 中                                                                          |

**适用场景：** 主题跳跃明显的复杂长文档；学术论文、法律文件；对检索质量要求高且有计算资源的离线场景

---

### 5.2 基于 LLM 的分块（LLM-Based Chunking）

**原理：**

直接让 LLM 分析文档内容并决定分割方式。LLM 可以识别文档中的逻辑结构、论证链条和主题边界，生成比规则方法更符合语义的分块。常见实现方式：

- 识别命题（将文本分解为独立的逻辑陈述）
- 按摘要粒度切分（让 LLM 生成各节摘要后基于摘要边界切分）
- 关键点提取（提取最重要信息作为独立 chunk）

**代码示例：**

```python
def llm_based_chunking(document: str, llm) -> list[str]:
    prompt = f"""
    请分析以下文档，将其分割成语义完整、逻辑连贯的片段。
    每个片段应：
    1. 聚焦于单一主题或论点
    2. 独立可读（无需前后文即可理解）
    3. 长度在 150-400 词之间

    用 <chunk> 和 </chunk> 标签标记每个片段。

    文档：
    {document}
    """
    response = llm.invoke(prompt)
    import re
    return re.findall(r'<chunk>(.*?)</chunk>', response, re.DOTALL)
```

**与 Agentic Chunking 的区别：**

|          | LLM-Based Chunking     | Agentic Chunking           |
| -------- | ---------------------- | -------------------------- |
| 决策内容 | 在哪里切分（边界位置） | 用哪种策略切分（策略选择） |
| 复杂度   | 较高                   | 极高（通常包含 LLM-Based） |
| 灵活性   | 固定提示词             | 动态决策                   |

**优缺点：**

| 维度 | 说明                                                          |
| ---- | ------------------------------------------------------------- |
| 优点 | 语义理解最深，能处理隐式逻辑结构                              |
| 缺点 | **成本高**（每文档需 LLM 调用）；速度慢；输出不稳定需额外解析 |
| 质量 | ★★★★★                                                         |
| 速度 | ★☆☆                                                           |
| 成本 | 高                                                            |

**适用场景：** 高价值文档（法律合同、医疗记录、监管文件）；预算充足的离线批处理

---

### 5.3 智能代理分块（Agentic Chunking）

**原理：**

使用 AI Agent 动态分析整个文档（包括结构、密度、内容类型），然后**自主选择最合适的分块策略组合**并执行。例如：Agent 发现文档是 Markdown 格式，则先按标题分割；发现某节内容密度高，则对该节再做语义分块；还可以自动为 chunk 打标签和 metadata。

**流程示意：**

```
文档输入
    ↓
Agent 分析（文档类型？结构？密度？）
    ↓
选择策略：Markdown → 按标题 | 密集段落 → 语义分块 | 表格 → 行转文本
    ↓
执行分块 + 生成 metadata
    ↓
质量验证（chunk 是否自包含？是否有孤立片段？）
    ↓
输出最终 chunks
```

**优缺点：**

| 维度 | 说明                                              |
| ---- | ------------------------------------------------- |
| 优点 | 质量最高；可针对每个文档的特点定制策略            |
| 缺点 | **成本极高**（多次 LLM 调用）；速度极慢；实现复杂 |
| 质量 | ★★★★★                                             |
| 速度 | ★☆☆                                               |
| 成本 | 极高                                              |

**适用场景：** 高价值少量文档；企业知识库中需要最高质量检索的核心文档

---

### 5.4 自适应分块（Adaptive Chunking）

**原理：**

使用 ML 模型分析文档不同段落的语义密度和结构复杂度，**动态调整** chunk_size 和 overlap 参数。与 Agentic Chunking 的区别：Adaptive 调整的是同一策略的参数，Agentic 选择的是不同策略。

**示例逻辑：**

```python
def adaptive_chunking(text: str, sections: list[str]) -> list[str]:
    chunks = []
    for section in sections:
        density = compute_semantic_density(section)  # 语义密度评分

        if density > 0.8:
            # 高密度段落：使用小 chunk，确保细粒度检索
            chunk_size, overlap = 256, 50
        elif density > 0.5:
            # 中等密度：标准参数
            chunk_size, overlap = 512, 80
        else:
            # 低密度（叙述性段落）：使用大 chunk 保留上下文
            chunk_size, overlap = 1024, 100

        section_chunks = fixed_size_chunking(section, chunk_size, overlap)
        chunks.extend(section_chunks)

    return chunks
```

**优缺点：**

| 维度 | 说明                                       |
| ---- | ------------------------------------------ |
| 优点 | 针对文档内容动态调优；比固定参数策略更精细 |
| 缺点 | 实现复杂；需要训练或调参的 ML 模型         |
| 质量 | ★★★★                                       |
| 速度 | ★★☆                                        |
| 成本 | 中高                                       |

**适用场景：** 结构和密度差异很大的混合文档集（同时包含技术文章、新闻、对话记录等）

---

## 六、第三类：结构增强分块方法

### 6.1 层次化分块（Hierarchical Chunking）

**原理：**

将文档分解为**多个粒度层级**，不同层级的 chunk 捕捉不同粒度的信息。检索时先在高层级定位大范围，再在低层级精确定位细节。

**层级结构示例：**

```
Level 1（标题/摘要，1000～2000 tokens）
    └── 第一章：RAG 系统概述
        Level 2（段落，300～500 tokens）
            └── 1.1 什么是 RAG
            └── 1.2 RAG 的应用场景
                Level 3（句子级，50～100 tokens）
                    └── RAG 通过检索外部知识...
                    └── 典型应用包括客服系统...
```

**代码示例（LlamaIndex）：**

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 三个层级的 chunk size
)

nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)  # 最小粒度的节点用于检索
```

**检索策略：**

- 用最小粒度（Level 3）做向量检索，精度高
- 找到命中节点后，向上取父节点（Level 2 或 Level 1）提供给 LLM，上下文丰富

**优缺点：**

| 维度 | 说明                                             |
| ---- | ------------------------------------------------ |
| 优点 | 同时支持高层摘要查询和细节查询；上下文丰富度可调 |
| 缺点 | 存储量成倍增加（多份同内容不同粒度）；实现复杂   |
| 质量 | ★★★★★                                            |
| 速度 | ★★☆                                              |
| 成本 | 中高                                             |

**适用场景：** 大型复杂文档（教材、法律合同、技术手册）；需要同时支持概述型和细节型查询

---

### 6.2 父子分块（Parent-Child Chunking）

**原理：**

层次化分块的简化版，只有两个层级：大块（父块）和小块（子块）。

- **子块**（小，200～400 tokens）：用于向量检索，语义聚焦、精度高
- **父块**（大，1000～2000 tokens）：用于 LLM 生成，上下文完整

**流程：**

```
索引阶段：
  文档 → 父块（存入文档存储）
           └── 子块（向量化，存入向量数据库，携带父块 ID）

检索阶段：
  查询 → 向量检索命中子块 → 根据父块 ID 取出父块 → 送入 LLM
```

**代码示例（LangChain）：**

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(documents)

# 检索时自动返回父块
results = retriever.get_relevant_documents("你的查询")
```

**优缺点：**

| 维度 | 说明                                                 |
| ---- | ---------------------------------------------------- |
| 优点 | 兼顾检索精度（小块）和生成质量（大块）；实现相对简单 |
| 缺点 | 存储量翻倍；父块选择不当会引入过多噪音               |
| 质量 | ★★★★★                                                |
| 速度 | ★★★                                                  |
| 成本 | 低（无需 LLM）                                       |

**适用场景：** 生产环境 RAG 系统的默认增强方案；各类文档通用

---

### 6.3 上下文增强分块（Contextual Chunking）

**原理（来自 Anthropic Contextual Retrieval 技术）：**

小 chunk 脱离原文后往往缺乏语义背景，导致 embedding 质量差和检索失败。Contextual Chunking 使用 LLM 为每个 chunk 生成一段简短的上下文描述，说明该 chunk 在文档中的位置和作用，并将描述**前置拼接**到 chunk 内容，使其在独立检索时仍有足够背景。

**示例对比：**

```
原始 chunk（缺乏上下文）：
"公司的净利润同比增长 23%，超出市场预期。"

上下文增强后：
"本文档是 2024 年年度财务报告，本段描述第三季度的核心财务指标。
公司的净利润同比增长 23%，超出市场预期。"
```

**代码实现：**

```python
import anthropic

client = anthropic.Anthropic()

def generate_contextual_chunk(document: str, chunk: str) -> str:
    prompt = f"""
    <document>
    {document}
    </document>

    以下是文档中的一个片段：
    <chunk>
    {chunk}
    </chunk>

    请用 2-3 句话说明这个片段在整个文档中的位置和作用，
    以便在片段被单独检索时能理解其上下文。只输出描述，不要其他内容。
    """
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    context = response.content[0].text
    return f"{context}\n\n{chunk}"
```

**效果（Anthropic 实验数据）：**

- 单独使用 Contextual Chunking：检索失败率降低 **49%**
- 结合 BM25 混合检索：进一步降低至 **67%**

**添加结构化 Metadata（补充方式）：**

```python
chunk_with_metadata = {
    "content": chunk_text,
    "metadata": {
        "document_title": "2024年度财务报告",
        "section": "第三季度核心指标",
        "page_number": 42,
        "source": "annual_report_2024.pdf",
        "chunk_id": "annual_report_2024_p42_c3"
    }
}
```

**优缺点：**

| 维度 | 说明                                                            |
| ---- | --------------------------------------------------------------- |
| 优点 | 从根本上解决小 chunk 缺乏上下文的问题；可与任意分块策略叠加使用 |
| 缺点 | 需要 LLM 调用，**预处理成本高**；每个 chunk 存储体积增大        |
| 质量 | ★★★★★                                                           |
| 速度 | ★★☆                                                             |
| 成本 | 高                                                              |

**适用场景：** 生产级 RAG 系统；文档中存在大量需要上下文才能理解的片段

---

### 6.4 命题分块（Proposition Chunking）

**原理：**

将文档分解为最小的**原子事实陈述单元**（命题），每个命题是一个独立的、语义自包含的知识点。

**示例：**

```
原文：
"爱因斯坦于1879年3月14日出生于德国乌尔姆，是相对论的创始人，
 曾于1921年获得诺贝尔物理学奖。"

命题分解：
- 爱因斯坦生于1879年3月14日。
- 爱因斯坦的出生地是德国乌尔姆。
- 爱因斯坦是相对论的创始人。
- 爱因斯坦于1921年获得诺贝尔物理学奖。
```

**使用 LLM 生成命题：**

```python
def extract_propositions(text: str, llm) -> list[str]:
    prompt = f"""
    请将以下文本分解为独立的事实陈述（命题）。
    每个命题应该：
    - 是一个完整的句子
    - 只包含一个事实
    - 语义自包含（无需上下文即可理解）

    每行输出一个命题。

    文本：{text}
    """
    response = llm.invoke(prompt)
    return [line.strip() for line in response.split('\n') if line.strip()]
```

**与父子分块结合（最佳实践）：**

```
命题（子块）→ 向量检索（精度极高）
原始段落（父块）→ 提供给 LLM（上下文完整）
```

**优缺点：**

| 维度 | 说明                                                        |
| ---- | ----------------------------------------------------------- |
| 优点 | 检索精度极高；每次命中都是最直接相关的具体事实              |
| 缺点 | 需要 LLM 生成，成本高；命题碎片化，需结合父子分块补充上下文 |
| 质量 | ★★★★★                                                       |
| 速度 | ★☆☆                                                         |
| 成本 | 高                                                          |

**适用场景：** 知识问答型 RAG（精确事实检索）；FAQ 系统；知识图谱构建

---

### 6.5 晚期分块（Late Chunking）

**原理（来自 Jina AI 的研究）：**

传统方式先分块再 embedding，各 chunk 的 embedding 相互孤立。晚期分块则**先对全文生成 token 级别的 embedding（保留全局上下文），再在 embedding 空间中按位置聚合**，使每个 chunk 的向量天然包含全文上下文信息。

**对比：**

```
传统方式：
  文本 → [分块] → Chunk1 → Embedding1（上下文孤立）
                 → Chunk2 → Embedding2（上下文孤立）

晚期分块：
  文本 → [全文 Embedding，产生 token 级向量]
       → [按位置分组，对组内 token 向量做 mean pooling]
       → Chunk1 Embedding（感知全局上下文）✓
       → Chunk2 Embedding（感知全局上下文）✓
```

**代码示例（使用 Jina Embeddings）：**

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "jinaai/jina-embeddings-v2-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def late_chunking(text: str, chunk_boundaries: list[tuple]) -> list:
    # 对全文做 token 级 embedding
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    # 按 chunk 边界聚合 token embeddings
    chunk_embeddings = []
    for start, end in chunk_boundaries:
        chunk_emb = token_embeddings[start:end].mean(dim=0)
        chunk_embeddings.append(chunk_emb)

    return chunk_embeddings
```

**优缺点：**

| 维度 | 说明                                                                |
| ---- | ------------------------------------------------------------------- |
| 优点 | chunk embedding 天然包含全局上下文；有效缓解"孤立 chunk"问题        |
| 缺点 | 需要支持长上下文的 embedding 模型；实现复杂；文档长度受模型上限限制 |
| 质量 | ★★★★                                                                |
| 速度 | ★★☆                                                                 |
| 成本 | 中                                                                  |

**适用场景：** 段落间有大量交叉引用的技术文档、法律文书、研究报告

---

## 七、分块参数详解

### 7.1 Chunk Size（块大小）

| 大小范围            | 特点                   | 适用场景                     |
| ------------------- | ---------------------- | ---------------------------- |
| 128～256 tokens     | 精确度高，上下文少     | 精确事实问答型 RAG           |
| **256～512 tokens** | **最常用的平衡范围**   | **通用文档检索（推荐起点）** |
| 512～1024 tokens    | 上下文丰富，精确度稍低 | 摘要型任务，长篇文章         |
| 1024+ tokens        | 接近全文检索           | 需要大量上下文的分析任务     |

**确定 chunk size 的迭代方法：**

1. 从 512 tokens + 10% overlap 开始作为基准
2. 跑评估集，测量 Recall@5 和 MRR
3. 若精度不足 → 缩小 chunk_size；若上下文不足 → 增大 chunk_size
4. 反复迭代，直到指标收敛

### 7.2 Embedding 模型与 Chunk Size 的匹配

chunk_size 应**小于** embedding 模型的最大输入长度，留有余量：

| Embedding 模型         | 最大 token | 建议 Chunk Size |
| ---------------------- | ---------- | --------------- |
| text-embedding-ada-002 | 8,191      | 256～512        |
| bge-large-zh           | 512        | 256～400        |
| bge-m3                 | 8,192      | 512～1024       |
| jina-embeddings-v2     | 8,192      | 512～2048       |

---

## 八、特殊文档类型的处理

### 8.1 PDF 文档

PDF 解析是分块前的关键步骤，常用工具：

| 工具           | 特点                 | 推荐场景         |
| -------------- | -------------------- | ---------------- |
| PyMuPDF (fitz) | 速度快，支持图片提取 | 通用 PDF         |
| pdfplumber     | 表格提取能力强       | 含大量表格的 PDF |
| PyPDF2         | 轻量，纯文本提取     | 简单文本 PDF     |
| Unstructured   | 多格式支持，结构感知 | 混合格式文档     |
| LlamaParse     | 基于 LLM，质量最高   | 复杂排版 PDF     |

> **最佳实践**：先将 PDF 转为 Markdown 格式（保留结构），再进行分块。

### 8.2 表格数据

表格不适合直接文本分块，应转换后处理：

```python
# 方式一：保留为 Markdown 表格整体向量化
table_markdown = """
| 姓名 | 年龄 | 职位 |
|-----|-----|-----|
| 张三 | 28 | 工程师 |
| 李四 | 32 | 经理 |
"""

# 方式二：每行转为自然语言描述（适合问答型 RAG，检索精度更高）
rows_as_text = [
    "张三，28岁，职位为工程师。",
    "李四，32岁，职位为经理。"
]
```

### 8.3 多模态文档（图文混排）

- 使用多模态模型（如 GPT-4V、Claude 3）对图片生成文字描述
- 将图片描述插入对应位置后再进行整体分块
- 或单独为图片建立向量索引，实现图文混合检索

---

## 九、分块策略全览对比

| 策略       | 类别 | 质量  | 速度 | 成本 | 适用场景         |
| ---------- | ---- | ----- | ---- | ---- | ---------------- |
| 固定大小   | 基础 | ★★☆   | ★★★  | 极低 | 快速原型         |
| 递归字符   | 基础 | ★★★   | ★★★  | 极低 | **通用首选**     |
| 文档结构   | 基础 | ★★★★  | ★★★  | 低   | 格式规范文档     |
| 代码分块   | 基础 | ★★★★  | ★★★  | 低   | 代码库           |
| 滑动窗口   | 基础 | ★★★   | ★★★  | 低   | 固定大小的增强   |
| 语义分块   | 智能 | ★★★★  | ★★☆  | 中   | 复杂非结构化文本 |
| LLM-Based  | 智能 | ★★★★★ | ★☆☆  | 高   | 高价值文档       |
| Agentic    | 智能 | ★★★★★ | ★☆☆  | 极高 | 关键文档离线处理 |
| 自适应     | 智能 | ★★★★  | ★★☆  | 中高 | 混合异构文档集   |
| 层次化     | 增强 | ★★★★★ | ★★☆  | 中高 | 大型复杂文档     |
| 父子分块   | 增强 | ★★★★★ | ★★★  | 低   | 生产环境通用增强 |
| 上下文增强 | 增强 | ★★★★★ | ★★☆  | 高   | 生产级 RAG       |
| 命题分块   | 增强 | ★★★★★ | ★☆☆  | 高   | 精确问答系统     |
| 晚期分块   | 增强 | ★★★★  | ★★☆  | 中   | 强交叉引用文档   |

---

## 十、工具库选型

| 工具库           | 定位            | 优势                            | 适合场景               |
| ---------------- | --------------- | ------------------------------- | ---------------------- |
| **LangChain**    | 全栈 LLM 框架   | TextSplitter 丰富，生态完整     | 构建完整 LLM 应用      |
| **LlamaIndex**   | RAG 专用框架    | NodeParser 高度优化，支持层次化 | 高性能数据检索系统     |
| **chonkie**      | 轻量分块专用库  | 专注分块，依赖少，接口简洁      | 只需分块功能的轻量项目 |
| **Unstructured** | 文档解析 + 分块 | 多格式支持，结构感知强          | 非结构化文档预处理     |

---

## 十一、最佳实践

### 11.1 分块完整流程

```
1. 文档解析（PDF/HTML/DOCX → 纯文本或 Markdown）
      ↓
2. 文本清洗（去除页眉页脚、乱码、多余空白）
      ↓
3. 选择分块策略（参考下方决策流程）
      ↓
4. 添加 Metadata（文档名、页码、章节、时间戳等）
      ↓
5. 评估分块效果（人工抽查 + 量化指标）
      ↓
6. 迭代优化（调整参数或切换策略）
```

### 11.2 策略选择决策流程

```
文档已经很短且完整？
  → 不需要分块，直接向量化

文档有明确结构（Markdown/HTML/代码）？
  → 文档结构分块 + 可选父子分块增强

普通非结构化文本？
  → 从递归字符分块开始（512 tokens, 10% overlap）

主题多样、跳跃明显？
  → 考虑语义分块

对检索质量要求极高且有预算？
  → 上下文增强分块（Contextual Retrieval）
  → 或 父子分块 + 命题检索

超大型复杂文档，需支持多层次查询？
  → 层次化分块
```

### 11.3 Metadata 设计

```python
chunk = {
    "content": "...",
    "metadata": {
        "source": "annual_report_2024.pdf",
        "page": 15,
        "section": "财务分析 > Q3 业绩",
        "created_at": "2024-01-15",
        "doc_type": "财报",
        "chunk_id": "annual_report_2024_p15_c3",
        "chunk_index": 15,      # 在文档中的序号（用于排序重建）
        "total_chunks": 120
    }
}
```

Metadata 的核心用途：

- **过滤检索**：按时间、来源、类型等缩小检索范围
- **上下文还原**：告知 LLM chunk 来自哪里
- **结果重排**：结合 metadata 做时效性、权威性排序

### 11.4 评估分块效果

**定性评估：**

- 人工抽样检查 50～100 个 chunk，确认语义完整
- 验证重要信息是否被边界切断

**定量评估指标：**

| 指标                                         | 说明                                            | 作用                                          | 公式/计算方式                                                                  |
| -------------------------------------------- | ----------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------ |
| Recall@K                                     | Top-K 检索结果中，相关 chunk 的召回率           | 衡量检索系统能否找到所有相关内容              | `Recall@K = (检索到的相关 chunk 数) / (所有相关 chunk 总数)`                   |
| MRR (Mean Reciprocal Rank)                   | 第一个相关结果的排名倒数均值                    | 评估相关结果的排名质量,越靠前越好             | `MRR = (1/N) * Σ(1/rank_i)`，其中 rank_i 是第 i 个查询首个相关结果的排名       |
| NDCG (Normalized Discounted Cumulative Gain) | 归一化折损累积增益，考虑排名位置的加权召回      | 综合评估检索结果的相关性和排序质量            | `NDCG@K = DCG@K / IDCG@K`，其中 `DCG = Σ(rel_i / log2(i+1))`，rel 为相关性分数 |
| Faithfulness                                 | LLM 生成答案与检索内容的一致性（用 RAGAS 评估） | 检测 LLM 是否基于检索内容回答，防止幻觉和编造 | 通过 LLM 判断生成答案中的每个陈述是否能从检索内容中推导出，计算支持率          |

**评估工具：**

- [RAGAS](https://github.com/explodinggradients/ragas)：RAG 专用评估框架，支持 Faithfulness、Answer Relevancy 等多维度
- [TruLens](https://github.com/truera/trulens)：LLM 应用评估和监控工具

### 11.5 常见问题排查

| 问题       | 现象                                     | 解决方案                                   |
| ---------- | ---------------------------------------- | ------------------------------------------ |
| Chunk 过小 | LLM 无法基于单个 chunk 回答，答案不完整  | 增大 chunk_size；使用父子分块              |
| Chunk 过大 | 检索精度低，Top-K 结果含大量无关内容     | 减小 chunk_size；使用语义分块              |
| 边界切断   | 关键信息被截断在两个 chunk 之间          | 增大 overlap；改用递归分块                 |
| 孤立 chunk | 小块脱离上下文后意义不明                 | 使用上下文增强分块（Contextual Retrieval） |
| 重复检索   | Top-K 结果内容高度相似                   | 减小 overlap；检索后 MMR 去重              |
| 冷数据浪费 | 大量文档从未被查询，但占用分块和存储资源 | 考虑 Post-Chunking 策略                    |

---

## 十二、面试高频问题

**Q1：RAG 中为什么需要分块？不能直接把整个文档向量化吗？**

> A：不能直接将整个文档向量化主要有三个原因：① **技术限制**：大多数 embedding 模型有 token 上限（如 512 或 8192），超长文档无法直接处理；② **语义稀释**：整篇文档包含多个主题，向量表示会"平均化"，检索时语义匹配精度下降；③ **上下文冗余**：即使 LLM 支持超长上下文，把大量无关文本传入也会增加成本并降低生成质量（Lost-in-the-Middle 问题）。

**Q2：如何选择合适的 chunk size？**

> A：需要综合考虑以下因素：① embedding 模型的最大输入长度（chunk size 应小于此值）；② 查询类型：精确问答型用小块（128-256 tokens），摘要分析型用大块（512-1024 tokens）；③ 文档特点：句子短的文档适合小块，长篇分析文档适合大块；④ 以 512 tokens 为起点，通过检索评估指标（Recall@K、MRR）迭代验证。

**Q3：什么是 Contextual Retrieval？它解决了什么问题？**

> A：Contextual Retrieval 是 Anthropic 提出的技术。传统分块的问题是：小 chunk 脱离原文后往往缺乏上下文，导致 embedding 质量差和检索失败。Contextual Retrieval 用 LLM 为每个 chunk 生成一段上下文描述（说明该 chunk 的位置和作用），并前置拼接到 chunk 内容，使每个 chunk 独立存在时仍有足够的语义背景。实验显示可将检索失败率降低约 **49%**，结合 BM25 混合检索可进一步降低至 **67%**。

**Q4：父子分块（Parent-Child Chunking）是什么原理，解决什么问题？**

> A：父子分块将文档切成两种粒度：**小块（子块，200～400 tokens）用于向量检索**（语义聚焦，精度高），**大块（父块，1000～2000 tokens）用于 LLM 生成**（上下文完整）。解决了"检索精度"和"生成质量"的两难困境：单纯用大块检索精度低，单纯用小块上下文不足。父子分块让检索和生成各取所需。

**Q5：语义分块相比递归字符分块有什么优劣？**

> A：**优势**：在语义层面理解文本，能在真正的"语义转折点"处切分，chunk 内部主题更连贯，适合主题跳跃明显的复杂文档。**劣势**：需要对每个句子生成 embedding，预处理时间和成本远高于递归字符分块；对于结构简单的文档，效果提升有限。**实践建议**：大多数场景递归字符分块是性价比最高的选择；仅在对质量要求极高且有计算资源时才考虑语义分块。

**Q6：晚期分块（Late Chunking）的核心思想是什么？**

> A：传统分块先切分再 embedding，各 chunk 的向量表示相互孤立，无法感知全文上下文（比如代词指代、跨段落引用）。晚期分块（Jina AI 提出）则先对全文做 token 级别的 embedding（保留上下文信息），再按位置对 token 向量做 mean pooling 得到各 chunk 的 embedding。结果是每个 chunk 的向量天然包含全局上下文，有效缓解了"孤立 chunk 语义丢失"的问题。

**Q7：Pre-Chunking 和 Post-Chunking 有什么区别？各适合什么场景？**

> A：**Pre-Chunking**（预分块）在索引阶段就完成分块和向量化，查询时直接检索，延迟低；适合绝大多数 RAG 场景。**Post-Chunking**（后分块）先对整文档向量化，只在检索命中后才对该文档实时分块；优点是不对冷数据浪费分块资源，支持查询感知的动态分块；适合文档库极大但查询集中在少量文档的场景，或需要根据查询动态调整分块策略的高级系统。

**Q8：如何处理表格、图片等非文本内容的分块？**

> A：① **表格**：将表格转为 Markdown 格式整体向量化；或将每行转为自然语言描述作为独立 chunk（问答精度更高）；② **图片**：使用多模态模型（如 GPT-4V、Claude 3）生成图片的文字描述，再对描述分块和向量化；也可单独为图片建立向量索引实现图文混合检索；③ **混合文档**：使用 Unstructured 或 LlamaParse 进行结构化解析，识别不同元素类型后分别处理。

---

## 参考资料

- [Weaviate: Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Firecrawl: Best Chunking Strategies for RAG 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Databricks: The Ultimate Guide to Chunking Strategies](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Pinecone: Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Coveo: RAG Chunking Information](https://www.coveo.com/blog/rag-chunking-information/)
- [IBM: Chunking Strategies for RAG with LangChain](https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai)
- [Cohere: Chunking Strategies](https://docs.cohere.com/page/chunking-strategies/)
- [MasteringLLM: 11 Chunking Strategies Simplified & Visualized](https://masteringllm.medium.com/11-chunking-strategies-for-rag-simplified-visualized-df0dbec8e373)
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Jina AI: Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
