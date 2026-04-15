# Structured Output Doc Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clarify the relationship between native structured outputs, schema-first libraries, and constrained decoding in `00_llm_basics.md`.

**Architecture:** Update the existing `Structured Output` section rather than adding a disconnected appendix. Keep the article general-purpose, using OpenAI-style APIs only as examples while clearly separating provider-native capabilities, application-layer schema libraries, and decoding-layer constraints.

**Tech Stack:** Markdown, Python examples, JSON Schema, Pydantic

---

### Task 1: Restructure the Structured Output section

**Files:**
- Modify: `Agent Guidance/Agent/00_llm_basics.md`

**Step 1: Rewrite the section outline**

Add a decision-oriented flow:
- JSON Mode
- Native Structured Outputs
- Schema-first libs
- Prompt-only fallback
- Relationship between the three concepts
- Practical selection guidance

**Step 2: Keep examples concrete**

Use short Python examples with Pydantic and provider APIs, but explain concepts in vendor-neutral language.

**Step 3: Strengthen conceptual boundaries**

Make explicit:
- Native structured outputs are provider/model capabilities
- Schema-first libs are developer-facing abstractions
- Constrained decoding is a lower-level generation control mechanism

**Step 4: Preserve article flow**

End the new material by reconnecting it to `JSON Schema` and the upcoming `Tool Calling` article.

### Task 2: Review for readability

**Files:**
- Modify: `Agent Guidance/Agent/00_llm_basics.md`

**Step 1: Check tone and transitions**

Ensure the new content matches the existing blog voice: explanatory, practical, and concise.

**Step 2: Check for redundancy**

Avoid repeating the same definition in both `Structured Output` and `JSON Schema`.

**Step 3: Final verification**

Read the updated markdown and confirm headings, examples, and transitions are coherent.
