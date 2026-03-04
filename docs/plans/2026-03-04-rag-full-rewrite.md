# RAG 文档体系重写 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 补齐 `06_evaluate.md`，并按 RAG 模块与流程重写 `Rag_Guidance.md` 与 `01~08`，删除重复、补齐缺失内容。

**Architecture:** 采用“总览 + 分模块”结构：
- `Rag_Guidance.md` 负责端到端流程、模块关系、阅读顺序。
- `01~08` 各文件仅覆盖单一模块，声明边界并交叉引用，避免重复。

**Tech Stack:** Markdown

---

### Task 1: 重写总览文档

**Files:**
- Modify: `Agent Guidance/RAG/Rag_Guidance.md`

**Step 1:** 定义统一流程图与模块职责
**Step 2:** 给出阅读路径和上线检查清单

### Task 2: 重写核心模块文档

**Files:**
- Modify: `Agent Guidance/RAG/01_embedding_model.md`
- Modify: `Agent Guidance/RAG/02_vector_search.md`
- Modify: `Agent Guidance/RAG/03_sparse_retrieval.md`
- Modify: `Agent Guidance/RAG/04_hybrid_search.md`
- Modify: `Agent Guidance/RAG/05_rerank.md`
- Modify: `Agent Guidance/RAG/06_evaluate.md`
- Modify: `Agent Guidance/RAG/07_query_enhancement.md`
- Modify: `Agent Guidance/RAG/08_context_enhancement.md`

**Step 1:** 统一每章结构（定位、影响、方法、参数、指标、风险）
**Step 2:** 删除跨章重复定义，改为链接引用
**Step 3:** 补齐空白与薄弱章节（尤其 02、05、06）

### Task 3: 验证

**Files:**
- Modify: 同上

**Step 1:** 检查 01~08 是否均非空且结构完整
**Step 2:** 检查 `Rag_Guidance` 链接与流程一致性
