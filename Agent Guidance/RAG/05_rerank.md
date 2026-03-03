ReRank
在召回阶段，我们通常得到几十到几百个候选文档，但它们的相关性可能参差不齐。
ReRank 的目标是对候选文档进行更精细的相关性排序，确保最相关的排在最前面。

原理
● 在召回阶段，我们只用轻量级的相似度（稀疏 or 稠密）快速找到候选文档。
● 在重排阶段，使用更复杂的模型 逐一对 Query 和候选文档打分，得到更精准的相关性排序。

常见方法

1. Cross-Encoder
   ○ 做法：将 Query 和 Document 拼接后一起输入到一个 Transformer（如 BERT、E5-Mistral、bge-reranker-large）。
   ○ 优势：模型能同时关注 Query 与 Document 的每个词，语义交互更细致。
   ○ 缺点：计算代价高，因为要对每个候选文档分别推理。
   ○ 应用：适合候选数目较少的场景（几十篇文档内）。
2. Bi-encoder + Cross-encoder 级联
   ○ 流程：
   i. 用 Bi-encoder（embedding 模型）做初步召回（几百篇）
   ii. 用 Cross-encoder 对 top-K（比如 50）做精排
   ○ 优势：兼顾效率和精度，工业界常用方案。

优点
● 提升检索精度（precision），保证最相关的文档排在前面。
● 尤其在召回结果包含噪音时，ReRank 能显著提升最终效果。
缺点
● 计算成本高（Cross-encoder 对每个候选文档都要跑一次模型）。
● 需要额外的高质量标注数据（query-文档相关性）来训练。
