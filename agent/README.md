# Agent Plan Workspace

本目录用于 Agent 化原型、验证脚本、文档和 wrapper 草稿。

- 不直接修改主产品现有源码目录（如 `core/`、`ui/`、`utils/`）
- 所有 Agent 相关实验先在本目录完成
- 通过验证后再决定是否迁移到正式实现位置

## 运行时 Agent 规范

- `D:\yolodo2.0\agent_plan\agent\AGENT.md`
  - 这是**运行时训练 Agent**自己的行为规范（参考 Claude Code 的规范化思路，但不是 Claude 专用文件）
  - 用来约束：
    - LLM 与代码分工
    - 工具选择原则
    - 上下文策略
    - 确认与失败处理
  - **不要把它和 Codex / 开发代理的仓库执行规则混淆**

