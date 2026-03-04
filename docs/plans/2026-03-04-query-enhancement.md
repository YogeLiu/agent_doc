# Query Enhancement 文档整理 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 新增系统化 Query Enhancement 文档，覆盖 Query Transformations、HyPE、HyDE，并补充对比场景；在现有 RAG 总览中增加索引入口。

**Architecture:** 采用“独立专题文档 + 主文档跳转索引”的结构。专题文档提供方法定义、流程、优缺点、失败模式与选型建议；主文档仅保留流程中的能力入口，避免重复。内容来源对齐用户给定 3 个 notebook，并补足工程落地细节。

**Tech Stack:** Markdown, 现有 RAG 文档结构（Agent Guidance/RAG）

---

### Task 1: 新建 Query Enhancement 专题文档

**Files:**
- Create: `Agent Guidance/RAG/07_query_enhancement.md`

**Step 1: 编写完整章节结构**
- 包含：背景、方法分类、Query Transformations（rewrite/step-back/sub-query）、HyDE、HyPE、组合策略、失败模式、评估与监控。

**Step 2: 增加方法对比与使用场景**
- 添加至少两张表：方法对比表、场景-推荐策略表。

**Step 3: 补充实现细节**
- 增加提示词约束、缓存策略、延迟与成本估算、常见参数建议。

### Task 2: 在主文档增加索引入口

**Files:**
- Modify: `Agent Guidance/RAG/vector_database.md`

**Step 1: 更新 RAG 链路描述**
- 将 Query Enhancement 显式命名并指向新文档。

**Step 2: 保持原有段落兼容**
- 不删除既有 HyDE 简介，仅增加“详见”链接，避免破坏已有阅读路径。

### Task 3: 自检

**Files:**
- Modify: `Agent Guidance/RAG/07_query_enhancement.md`
- Modify: `Agent Guidance/RAG/vector_database.md`

**Step 1: 一致性检查**
- 术语统一：Query Transformation / HyDE / HyPE。

**Step 2: 可读性检查**
- 标题层级和表格渲染正常，中文术语准确。
