# Agent Client 重构报告（2026-04-14）

> **当前状态：仍有效，但定位是“重构判断依据”，不是“当前代码逐行导航图”。**
>
> 这份文档最有价值的部分是：
>
> - 为什么问题重点在 `agent/client`
> - 为什么要推 LLM-first / tool-grounded
> - 为什么要压缩 pre-LLM code intercept 和 narrative-heavy fallback
>
> 需要注意：
>
> - 文中提到的部分函数行号、位置描述会随着后续提交漂移
> - 判断当前 runtime 接口与审批契约时，应优先同时参考 `docs/agent-runtime-contract.md`

## 1. 结论

当前问题不在 `agent/server` 的训练/预测/数据工具层，而在 `agent/client` 的编排层。

现状是：

- `agent/client/agent_client.py` 在进入 LangGraph 之前，已经用大量 `_try_handle_*` 把自然语言主路由截流。
- `agent/client/context_builder.py` 和 `agent/client/event_retriever.py` 会向模型灌入过重且容易串味的状态。
- `agent/client/grounded_reply_builder.py` 直接替模型组织最终回复，导致 Agent 退化成“规则壳 + 模板机”。

因此，当前系统的核心问题不是“模型不够强”，而是：

> client 抢走了模型本应承担的意图理解、工具选择和自然语言组织职责。

本次建议的重构方向不是推倒重写，而是：

> **保留 `agent/server` 工具与训练底座，重构 `agent/client` 为“模型优先、代码兜底”的 Agent 编排层。**

---

## 2. 与官方模式的对照结论

### 2.1 OpenAI

OpenAI 当前公开资料更接近：

- 用 **capable model + well-defined tools + clear structured instructions** 作为核心三件套；
- 用 guardrails / approvals / evals 做分层约束；
- 先做简单工作流，再逐步演进，而不是把自然语言主入口写成大量手工路由。

参考：

- [OpenAI – A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [OpenAI – Agent Builder guide](https://developers.openai.com/api/docs/guides/agent-builder)
- [OpenAI – Safety in building agents](https://platform.openai.com/docs/guides/agent-builder-safety)

### 2.2 LangChain / LangGraph

LangChain / LangGraph 当前公开资料更接近：

- 让模型承担工具调用；
- 用 middleware 在 `before_model` / `after_model` / `wrap_tool_call` 等阶段做校验、审批和错误处理；
- 按任务动态压缩上下文，而不是把整份会话状态持续塞给模型。

参考：

- [LangChain – Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain – Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [LangChain – Human in the loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [LangChain – Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)

### 2.3 Anthropic

Anthropic 当前公开资料更接近：

- 通过高质量工具定义、`input_schema`、`input_examples` 帮助模型正确选工具；
- 默认 `tool_choice=auto`，只在必要时强制某个 tool；
- 明确建议：返回高信号、低噪声的工具结果，而不是把杂乱大对象直接塞回上下文。

参考：

- [Anthropic – Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)

### 2.4 对照结论

当前项目与官方推荐模式的偏差点不是“用了 LangGraph 还不够”，而是：

- 过度前置硬路由；
- 过度 session fallback；
- 过度模板化确认与回复；
- 过度状态注入；
- 过度由代码替模型说话。

---

## 3. 当前代码中的主要偏航点

### 3.1 `agent_client.py` 把 LangGraph 架空了

文件：

- `agent/client/agent_client.py`

关键位置：

- `SYSTEM_PROMPT`：当前约第 130 行
- `chat()`：当前约第 956 行
- `_try_handle_mainline_intent()`：当前约第 1986 行
- `_try_handle_guardrail_intent()`：当前约第 2018 行
- `_build_training_plan_draft()`：当前约第 4647 行
- `_try_handle_training_plan_dialogue()`：当前约第 5092 行
- `_build_confirmation_prompt()`：当前约第 5133 行

补充说明：

- 旧文中的 `_try_handle_prepare_only_intent()` 已不再是当前代码里的独立前置入口，可视为已被并入/消化到更新后的 mainline / training-plan 流程中。

现状：

- 绝大多数自然语言在到 `self.graph.ainvoke(...)` 之前，就被 client 层人工分流；
- 模型并没有真正主导工具选择；
- 训练、prepare、predict、upload 被写成了“代码先猜一遍”的流程。

### 3.2 状态注入过重且容易串味

文件：

- `agent/client/context_builder.py`
- `agent/client/event_retriever.py`

现状：

- 每轮都给模型追加大块“当前结构化上下文”；
- 还会喂最近工具结果、确认记录、历史摘要；
- 对 `gemma4:e4b` 这类有限上下文模型来说，这不是“增强”，而是污染。

### 3.3 `grounded_reply_builder.py` 替模型说话

文件：

- `agent/client/grounded_reply_builder.py`

现状：

- `training_readiness`
- `check_training_status`
- `start_training`
- `predict_images`
- `upload_assets_to_remote`

这些工具都被代码直接拼成最终中文回复。

结果：

- 模型的作用被压缩成“有时选工具”；
- 回复风格被写死；
- 一旦工具设计不合理，回复也会跟着错误固化。

### 3.4 工具语义被混用

例如：

- “帮我看一下这份数据能不能直接训练”

当前被路由成：

- `training_readiness`

但 `training_readiness` 当前把：

- `data.yaml`
- 标签问题
- GPU 空闲
- auto device 解析

全绑在一起判断。

这会把一个**数据集可训练性问题**错误升级成**训练执行 readiness 问题**。

---

## 4. 当前必须承认的设计问题

### 4.1 这层现在不算合格 Agent

原因：

- 模型没有主导意图理解和工具选择；
- 代码大量替模型作答；
- 用户不得不迁就系统内部工具逻辑。

### 4.2 当前上下文策略不正确

正确方式应当是：

- 只给当前任务所需的最小上下文；
- 只有明确“刚才 / 继续 / 恢复 / 上次”时，才复用旧状态；
- 工具结果回流也应该是高信号、短结构，而不是整个大对象。

### 4.3 当前工具语义边界不清

必须拆开：

- **dataset-only readiness**
  - 只回答：数据结构、标签、split、`data.yaml`
- **execution readiness**
  - 才回答：GPU、device、environment、preflight、model

如果这两者不拆，client 就会继续把“看数据”误导成“准备开训”。

---

## 5. 正确的目标架构

## 5.1 职责划分

### 模型负责

- 意图理解
- 工具选择
- 参数补全建议
- 基于工具结果组织自然语言回复

### 代码负责

- 路径 / 文件存在性校验
- 高风险操作确认
- Tool schema / result 校验
- Session 持久化
- Interrupt / resume
- 错误兜底

## 5.2 对话主链应改成

1. 用户输入自然语言
2. 代码只做输入级 guardrail（空输入、slash command、显式审批态）
3. 模型读取：
   - 当前用户消息
   - 工具 schema / description
   - 最小上下文
4. 模型选择 tool
5. 代码执行 tool，并只回流高信号结构化结果
6. 模型基于工具结果组织回复
7. 若命中高风险 tool，再走 approval

---

## 6. 重构方案

## Phase 1：把“代码先替模型决定”改成“模型先决定”

### 目标

减少 `agent_client.py` 前置 `_try_handle_*` 的覆盖面积。

### 动作

保留：

- slash command
- 高风险确认
- 极少数系统 guardrail
- 输入合法性硬校验

砍薄：

- `_try_handle_training_plan_dialogue`
- `_try_handle_mainline_intent`

补充说明：

- `_try_handle_prepare_only_intent` 作为独立前置分支已经不是当前代码主干的一部分，因此后续重点不再是“删掉这个旧函数名本身”，而是继续减少剩余 pre-LLM intercept 的职责和覆盖面。

原则：

- 除非是高风险执行、明确命令态、或者硬错误拦截；
- 否则先交给模型做 tool selection。

### 验收

用户不需要记工具名；
自然话不再先被 if/else 截流。

---

## Phase 2：重做上下文策略

### 目标

避免上下文爆炸与串味。

### 动作

重构：

- `agent/client/context_builder.py`
- `agent/client/event_retriever.py`

默认只给模型：

- 当前用户输入
- pending confirmation
- 当前活跃对象的最小摘要
- 最近 1~2 条必要工具结果

只有显式提到：

- 刚才
- 上次
- 继续
- 恢复
- 沿用

才加载更老状态。

### 验收

- `/status` 不再污染下一句
- 新 session 默认轻启动
- 模型上下文显著变短

---

## Phase 3：把回复组织权还给模型

### 目标

`grounded_reply_builder.py` 不再替模型直接生成最终中文回复。

### 动作

将 `grounded_reply_builder.py` 改成：

- **事实压缩器**
- **证据摘要器**

只输出：

- `facts`
- `warnings`
- `blockers`
- `next_actions`
- `resolved_paths`

不直接拼“当前判断 / 当前阻塞 / 下一步建议”这种完整回答。

然后把这些事实作为 tool result 的高信号摘要回流给模型。

### 验收

模型的回复变自然；
代码只负责“喂事实”，不负责“替模型写答案”。

---

## Phase 4：拆开 dataset readiness 与 execution readiness

### 目标

避免“只看数据”时错误引入 GPU / 环境 / 模型问题。

### 动作

新增或重构成两个工具语义：

1. `dataset_training_readiness`
   - 检查：
     - images / labels
     - 标签质量
     - 是否已 split
     - 是否有可用 `data.yaml`

2. `training_execution_readiness`
   - 检查：
     - model
     - training environment
     - GPU / device
     - preflight

原则：

- `帮我看这份数据能不能直接训练`
  - 只走 dataset-only
- `现在能不能直接开训`
  - 才走 execution readiness

### 验收

数据问题只返回数据问题；
开训问题才返回 GPU / environment / model。

---

## Phase 5：确认页只基于已核实事实

### 目标

杜绝“没核实就先给一页看起来像真的确认”。

### 动作

所有 confirm 前必须满足：

- 路径存在
- 工具前置结果为 `ok`
- 必要字段存在

确认页只展示：

- 已核实到的路径
- 已核实到的阻塞项
- 将执行的具体动作

### 验收

错路径不能再进入 confirm；
prepare-only 不再混进训练阻塞。

---

## 7. 执行顺序建议

### P0

1. Phase 4：先拆 `dataset readiness` / `execution readiness`
2. Phase 1：砍前置硬路由

原因：

- 这是当前语义最错的地方；
- 不先拆开，后面所有回答都会继续混。

### P1

3. Phase 2：重做上下文注入
4. Phase 3：把回复组织权还给模型

### P2

5. Phase 5：统一确认页和审批流

---

## 8. 每一步怎么验证

### 验证句 1

```text
帮我看一下 /home/kly/ct_loop/data_ct 能不能直接训练
```

期望：

- 只回答数据集问题；
- 不扯 GPU；
- 不扯训练环境；
- 不扯模型；
- 不泄露工具名。

### 验证句 2

```text
用 /home/kly/ct_loop/data_ct 按默认比例准备训练数据，先不要开始训练。
```

期望：

- 只进入 prepare；
- 不生成训练计划草案；
- 不出现 GPU / 模型阻塞。

### 验证句 3

```text
/status
```

期望：

- 默认只返回高信号摘要；
- 不再吐整段 JSON；
- 不污染下一句自然语言判断。

### 验证句 4

```text
帮我把这个模型和数据上传到服务器
```

期望：

- 模型自己决定是否走上传工具；
- 如果信息不全，先追问必要缺口；
- 不默认偷吃旧路径。

---

## 9. 风险与边界

### 9.1 不是“全部交给模型”

高风险动作仍必须保留：

- confirm
- file existence check
- destructive tool guardrail
- interrupt / resume

### 9.2 不是重写 server

当前 server 工具层大体可保留：

- `agent/server/tools/*.py`
- `agent/server/services/*.py`

重点是把 client 改薄。

### 9.3 不是取消结构化状态

状态仍要保留，但：

- 默认不灌给模型；
- 只在明确 follow-up 时按需加载。

---

## 10. 最终判断

当前项目**不是底座全废了**，而是 **`agent/client` 这一层走偏了**。

正确做法不是继续修补模板壳，而是：

> **把 client 重构成真正的 Agent 编排层：模型优先、代码兜底、上下文最小化、工具语义拆清、确认基于已核实事实。**

如果不做这次重构，这个 Agent 继续往前加功能，只会越来越像一套“没有 UI 的规则 app”，而不是一个可用的 Agent。

---

## 11. 可复用框架 / 设计方案筛选

本节只看外部方案，不参考我们自己的 GitHub 代码。

### 11.1 方案 A：继续保留 LangGraph / LangChain，但按官方模式重构（推荐）

适配度：**最高**

原因：

- 我们现有项目已经使用 LangGraph；
- Python 技术栈一致；
- 已有 MCP tools / training tools / loop service 可以直接复用；
- LangChain 官方已有：
  - middleware
  - human-in-the-loop
  - context engineering
  - interrupt / resume

最适合复用的不是“把现在的 client 修修补补”，而是直接按 LangChain 官方推荐模式重排：

1. 使用 `create_agent` / LangGraph 作为主入口；
2. 用 middleware 做：
   - 高风险审批
   - tool error handling
   - before_model / after_model 过滤
3. 用最小上下文策略替换当前的重状态注入；
4. 让模型主导 tool selection；
5. 让 server 继续保留为工具执行层。

对应官方依据：

- LangChain Agents
- LangChain custom middleware
- LangChain human-in-the-loop
- LangChain context engineering

### 11.2 方案 B：部分借鉴 OpenAI Agents SDK 的结构，但不直接迁移（借鉴，不推荐直接替换）

适配度：**中等**

优点：

- OpenAI Agents SDK 在“agent definitions / running agents / guardrails / results and state / eval”这套结构上非常清晰；
- 很适合作为我们整理 client 分层职责的参考模板。

缺点：

- 当前项目必须兼容：
  - Ollama
  - OpenAI-compatible / API key 入口
  - 本地/服务器混合部署
- 如果直接切到 OpenAI Agents SDK，会带来 provider 和运行时适配成本；
- 对现有 MCP / 自定义 client 交互链未必最省改造成本。

因此更适合：

> **拿 OpenAI Agents SDK 当结构参考，不拿它当当前项目的直接替换框架。**

### 11.3 方案 C：PydanticAI 作为“轻量 Agent 运行时”候选（备选）

适配度：**中等偏高**

官方资料显示：

- Pydantic AI 是 model-agnostic；
- 支持多个 provider；
- OpenAI-compatible providers 中明确包含 Ollama；
- 还自带 eval / test model / MCP 支持入口。

优点：

- 比当前 client 自写编排更干净；
- 更容易把“工具 schema + 结构化输出 + provider 适配”统一起来；
- 对我们这种“工具很多，但想把入口做薄”的项目有吸引力。

缺点：

- 迁移成本高于“继续用 LangGraph 重构 client”；
- 需要重新组织 agent loop、审批流和状态流；
- 现有 LangGraph / MCP 代码不能无缝原样复用到入口层。

因此结论是：

> **PydanticAI 可以作为备选技术路线，但不是当前第一优先级。**

### 11.4 方案 D：Anthropic 的 tool-use 设计作为 tool schema 设计规范（借鉴）

适配度：**高，但不是框架**

Anthropic 官方最值得借鉴的不是运行时，而是：

- tool description 要足够细；
- `input_schema` 要严；
- `input_examples` 可帮助工具调用稳定；
- 默认 `tool_choice=auto`；
- tool result 应该是高信号、低噪声。

这很适合我们当前项目，因为我们现有大量工具定义虽然可用，但：

- 描述不够统一；
- 语义边界没拆清；
- result payload 过重。

因此 Anthropic 更像：

> **工具定义规范参考，而不是当前直接替换的运行时框架。**

---

## 12. 选择建议

### 推荐路线

**主路线：方案 A**

> **继续保留 LangGraph / LangChain 运行时，重构 `agent/client`，严格按 middleware + HITL + 最小上下文模式收口。**

### 辅助借鉴

- 用 **OpenAI Agents SDK** 做分层设计参考；
- 用 **Anthropic tool definition** 规范重写工具描述和输入 schema；
- 把 **PydanticAI** 作为中期备选，如果后面确认 LangGraph 入口层仍然太重，再评估迁移。

### 不推荐的路线

当前不建议：

- 直接切 OpenAI Agents SDK 作为唯一运行时
- 现在就整体迁移到 PydanticAI
- 继续在现有 `agent_client.py` 上堆更多 if/else

---

## 13. 重构落地策略（结合可复用方案）

### 第一阶段：保留 LangGraph，只替换用法

目标：

- 不动 server tool 层；
- 不动训练/预测/loop 服务层；
- 只把 client 从“规则壳”改成“模型优先的 orchestrator”。

动作：

1. 把 `chat()` 的前置硬路由砍薄；
2. 把审批改到 middleware / interrupt 思路；
3. 把 context_builder / event_retriever 改成最小上下文；
4. 把 `grounded_reply_builder` 降级成“事实摘要器”；
5. 把 dataset-only readiness 从 execution readiness 拆开。

### 第二阶段：工具定义重写

目标：

- 让模型更稳定地自动选对工具。

动作：

1. 重写 tool descriptions；
2. 为复杂 tool 增加输入样例；
3. 收缩 tool responses，只保留高信号字段；
4. 清理名称歧义和历史别名污染。

### 第三阶段：再决定是否替换运行时

如果完成第一阶段后：

- LangGraph client 仍然太重，
- 代码量仍然主要花在 agent loop 自己维护上，

再评估是否迁移到 PydanticAI 这类更轻量入口层。

