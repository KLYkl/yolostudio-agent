# Docs 状态说明

## 当前有效文档

- `/absolute/path/D:/yolodo2.0/agent_plan/docs/agent-client-refactor-report.md`
  - 当前 Agent client 重构基线
  - 用来指导现在的 LLM-first / tool-first 改造

## 已归档（pre-LLM / 旧 client 强路由阶段）

这些文档仍保留做历史追溯，但**不再作为当前设计或测试依据**：

- `/absolute/path/D:/yolodo2.0/agent_plan/docs/archive/pre-llm-agent/agent-training-loop-v1-flow.md`
- `/absolute/path/D:/yolodo2.0/agent_plan/docs/archive/pre-llm-agent/agent-training-loop-v1-test-plan.md`

## 使用原则

- 如果你要判断“当前系统应该怎么改 / 怎么测”，先看当前重构报告。
- 如果你要追溯“过去为什么这么做”，再去看 archive。
- 不要把 archive 里的旧验证和旧流程，当成当前实现已经满足的事实。
