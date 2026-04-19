# Docs 状态说明

本目录现在同时承载三类文档：

1. **当前仍可直接指导开发/联调的文档**
2. **有参考价值但偏历史快照的过程文档**
3. **明确归档的 pre-LLM / 旧强路由阶段文档**

## 当前有效文档

### `docs/agent-runtime-contract.md`
- 当前最接近“运行时接口契约”的文档
- 适合回答：
  - runtime 对外暴露哪些接口
  - `pending_action` / `needs_confirmation` 结构是什么
  - shell / runtime / HITL / tool layer 的边界怎么分
- 如果你要判断 **现在代码应该如何接 runtime**，优先看它

### `docs/agent-client-refactor-report.md`
- 当前 client 重构的背景、问题判断和目标方向
- 适合回答：
  - 为什么重点改 `agent/client` 而不是 `agent/server`
  - 为什么要推 LLM-first / tool-grounded
  - 为什么要缩减 pre-LLM code intercept 和 narrative-heavy fallback
- 注意：
  - 这份文档的**方向判断仍有效**
  - 其中部分**代码行号、函数位置和局部实现描述会随代码演进漂移**
  - 它更适合作为“设计依据”，不适合作为“逐行代码导航图”

### `knowledge/README.md`
- 当前知识来源、source policy 和防污染边界说明
- 适合回答：
  - 默认允许哪些知识源
  - `case` / `test` 为什么默认不直接参与正式建议
  - 为什么测试结果不能直接变成经验库

## 历史过程文档（可参考，但不是当前真相源）

### `docs/evaluation-driven-plan-adjustment-20260416.md`
- 这是 2026-04-16 收口阶段的评估/验收记录
- 适合回答：
  - 当时为什么这样收口
  - 哪些 direct-tool / grounded / intercept 问题被重点追踪过
  - 某批重构是按什么口径验收的
- 不适合直接当作：
  - 当前代码结构说明书
  - 当前唯一测试基线
  - 当前函数分布导航图

## 已归档（pre-LLM / 旧 client 强路由阶段）

以下内容只保留给历史追溯，**不再作为当前设计或测试依据**：

### 归档提示页
- `docs/agent-training-loop-v1-flow.md`
- `docs/agent-training-loop-v1-test-plan.md`

### 历史原文
- `docs/archive/pre-llm-agent/agent-training-loop-v1-flow.md`
- `docs/archive/pre-llm-agent/agent-training-loop-v1-test-plan.md`

## 使用原则

- 如果你要判断“**现在系统应该怎么接 / 怎么改 / 怎么测**”，先看：
  1. `docs/agent-runtime-contract.md`
  2. `docs/agent-client-refactor-report.md`
  3. `knowledge/README.md`
- 如果你要追溯“**某轮重构当时为什么这么收口**”，再看：
  - `docs/evaluation-driven-plan-adjustment-20260416.md`
- 如果你要追溯“**旧强路由阶段当年的设想**”，再看 archive。
- 不要把历史收口记录或 archive 文档，当成当前实现已经满足的事实。
