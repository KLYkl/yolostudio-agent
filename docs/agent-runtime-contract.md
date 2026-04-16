# Agent Runtime Contract

更新时间：2026-04-15

## 1. 目标

定义 `D:\yolodo2.0\agent_plan\agent\client` 的运行时边界，确保：

- CLI 只做 chat shell
- runtime 负责 model 驱动的 workflow continuation
- 高风险动作统一通过 structured interrupt / HITL 处理
- tool facts 与最终自然语言答复解耦

补充稳定约束（2026-04-15）：

- 配置默认值应优先尊重 provider 官方默认行为
- 非必要不覆盖 provider 官方行为
- 如确需偏离官方默认，必须：
  - 先证明该变更是通用方案，而不是针对单台机器/单个模型的特例
  - 以显式配置开关承载，而不是偷偷改成全局默认
  - 先得到用户同意，再作为默认策略落地
- 任何与 provider / runtime 默认值相关的修复，都必须优先参考官方文档，而不是基于当前服务器现象拍板

---

## 2. 分层边界

### 2.1 CLI / Chat Shell

CLI 只负责：

- 收集用户输入
- 打印流式 token / tool 事件
- 绑定 session / thread
- 调用 runtime API（`chat(...)`）
- 处理退出、编码、EOF、终端异常

CLI 不负责：

- 业务路由
- 直接拼装业务答案
- 审批关键词解析
- 直接调用高风险 tool shortcut
- 替 model 决定 prepare / train / loop 顺序

### 2.2 Agent Runtime

runtime 负责：

- 组织最小上下文
- 调 model 做 tool selection / continuation
- 把高风险 tool call 转成 structured interrupt
- 解释用户对 pending action 的自然语言回复
- 把自然语言审批归一化成 structured decision

### 2.3 Middleware / HITL

当前项目里暂由 runtime 内部承接，职责包括：

- 标记高风险 tool
- 生成 `pending_action`
- 接受 approve / reject / edit / clarify
- 持久化 pending state
- 恢复或取消执行

### 2.4 Tool Layer

tool 层只返回高信号事实，不承担最终 UI 文案职责。

---

## 3. Runtime 输入输出契约

## 3.1 `chat(user_text, ...)`

输入：

- `user_text: str`
- `stream_handler: optional`

输出至少包含：

- `status`
- `message`
- `tool_call`（如有）

状态枚举：

- `completed`
- `needs_confirmation`
- `cancelled`
- `error`

### 3.1.1 `needs_confirmation` 额外字段

当 runtime 命中高风险动作时，必须返回：

```json
{
  "status": "needs_confirmation",
  "message": "...",
  "tool_call": {
    "name": "start_training",
    "args": {}
  },
  "thread_id": "session-turn-3",
  "pending_action": {
    "interrupt_kind": "tool_approval",
    "decision_state": "pending",
    "thread_id": "session-turn-3",
    "tool_name": "start_training",
    "tool_args": {},
    "summary": "...",
    "objective": "...",
    "allowed_decisions": ["approve", "reject", "edit", "clarify"],
    "review_config": {
      "risk_level": "high",
      "allow_edit": true,
      "allow_clarify": true,
      "tool_name": "start_training"
    },
    "decision_context": {}
  }
}
```

约束：

- `pending_action` 是 shell/UI 唯一依赖的审批对象
- shell 不自己解释审批语义
- UI 可以把 `summary / objective / allowed_decisions / review_config` 直接展示给用户

## 3.2 `get_pending_action()`

输入：无

输出：

- 当前 pending action 的结构化对象
- 如无 pending，返回 `None`

用途：

- shell 恢复
- UI 恢复
- 测试验证

## 3.3 `review_pending_action(decision_payload, ...)`

输入：

```json
{
  "decision": "approve|reject|edit|clarify|restate|unclear",
  "reason": "...",
  "raw_user_text": "...",
  "source": "natural_language_chat|ui|api",
  "edits": {}
}
```

当前保证：

- `approve` -> 执行 pending action
- `reject` -> 取消 pending action
- `restate` -> 重新输出确认说明
- `unclear` / `clarify` / `edit` -> 保留 pending action，并返回自然语言引导用户继续说明

说明：

- 这允许 shell/UI 与自然语言审批解耦
- 自然语言聊天入口会先把用户输入归一化，再调用这个 runtime API

## 3.4 `confirm(thread_id, approved, ...)`

这是历史兼容 API。

保留原因：

- 已有测试
- 已有远端脚本
- 已有内部 follow-up 流程

定位：

- 兼容层
- 不再是主交互 contract

---

## 4. Pending Action 契约

字段说明：

- `interrupt_kind`
  - 当前固定为 `tool_approval`
- `decision_state`
  - `pending | approved | rejected`
- `thread_id`
  - 恢复该动作所需 thread
- `tool_name`
  - 待审批工具名
- `tool_args`
  - 结构化工具参数
- `summary`
  - 给用户看的简短动作摘要
- `objective`
  - 当前动作的业务目标
- `allowed_decisions`
  - UI/测试可依赖的允许动作集合
- `review_config`
  - 风险等级与 review 能力
- `decision_context`
  - 最近一次审批解释结果，例如：
    - decision
    - reason
    - raw_user_text
    - source
    - edits

---

## 5. 自然语言审批契约

自然语言输入仍然走 `chat(...)`，但 runtime 必须把它归一化成 structured decision。

例如：

- `继续`
  - -> `approve`
- `不继续`
  - -> `reject`
- `再说一遍`
  - -> `restate`
- `把 batch 改成 12 再继续`
  - -> 当前先视为 `edit/unclear`，保持 pending，不得误执行

重要约束：

- shell 不解释这些语义
- runtime 可以使用 model + fallback 归一化
- 不能只依赖 CLI 固定词表

---

## 6. 流式事件契约

当前 shell 依赖的流式事件类型：

- `token`
- `tool_call`

约束：

- shell 只渲染，不改变业务状态
- shell 不把 tool_call 自动当作批准

---

## 7. 设计约束

### 7.1 可以留在代码层的逻辑

- 空输入
- 文件/路径存在性硬校验
- 高风险 tool 标记
- interrupt persistence / resume
- 明确 destructive guardrail

### 7.2 不应留在 shell 层的逻辑

- slash business commands
- 审批固定词表解释器
- prepare/train/loop 主路线选择
- 最终业务答复模板拼装

### 7.3 不应继续扩张的逻辑

- 大量 `_try_handle_*` 抢 model 主路由
- 通过 regex 把用户话术硬恢复成业务流程
- 把工具结果直接写成终态中文说明模板

---

## 8. 当前落地状态（2026-04-15）

已完成：

- CLI 已退回薄壳
- `Confirm:` 特殊模式已移除
- pending confirmation 已有结构化 `pending_action`
- runtime 已提供：
  - `get_pending_action()`
  - `review_pending_action(...)`
- `confirm(...)` 已退化为兼容层

待继续：

- 把更多前置硬路由从 `agent_client.py` 下沉
- 把 edit / clarify 从“保留 pending”升级成真正 structured revision flow
- 继续压缩上下文注入
- 继续收口 tool result facts contract
