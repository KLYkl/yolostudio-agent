# 2026-04-16 评估驱动的收口计划修正

## 目的
基于最新 commit_progress_evaluation 审查结论，修正后续收口优先级，避免继续停留在“下沉 ownership 但全仓 code-first 不真实下降”的局面。

## 修正后的后续优先级
1. **真实删除 helper 中已被 classifier + cached state 覆盖的旧 direct-tool 分支**
   - 后续不再以“继续下沉”作为主要进度指标。
   - 必须观察 helper 内 `_complete_direct_tool_reply` 是否真实减少。

2. **缩减 `chat()` 前置 code 拦截层**
   - 重点审视 `_try_handle_prepare_only_intent` 与 `_try_handle_training_plan_dialogue` 的职责边界。
   - 方向是减少 pre-LLM code intercept 层数，而不是继续在外层包更多路由分支。

3. **继续减少 `agent_client.py` 对 `intent_parsing.py` 的消费端依赖**
   - parsing 层目标是纯参数提取，不再承载旧式语义判断。
   - 后续不仅看 `intent_parsing.py` 文件变小，也看导入/调用是否真实减少。

4. **推进 `grounded_reply_builder.py` 退出主链**
   - 保持 facts-first fallback，不扩 narrative-heavy 逻辑。
   - 后续应逐步缩小它在主链中的影响面。

## 保持不变的边界
- native structured output 仍只对已确认稳定的 `ollama` 开启；其他 provider 保持 fallback。
- 不为追求“更纯”引入兼容性风险。
- 继续遵守：model 负责 planning/orchestration，code 只负责 guardrail/state/execution。

## 测试约束
- 不以“主路由内部归零”替代“全仓真实删减”的结论。
- 每次删减后继续做跨域回归：
  - prediction / dataset / extract
  - training / knowledge / history / loop-history
  - realtime / remote
  - pending / dialogue / roundtrip
- 若缓存快捷路径导致跨域语义回归，必须当轮回退，不保留局部最优改动。

## 当前执行口径
后续进度汇报以“当前阶段进度”为主，详细实施细节继续写入仓内文档，不在对话里展开。


## 进度补充（2026-04-16）
- 已开始落实“推进 `grounded_reply_builder.py` 退出主链”的评估意见：tool-result fallback 在 planner 不可用时也优先消费 structured facts，再退回 grounded builder。
- 已继续落实“压缩 `chat()` 前置 code 拦截层”的评估意见：guardrail 前置拦截已并入 mainline router，不再单独占据 `chat()` 顶层分支。
- 已开始落实“真实减少 helper 中旧 direct-tool 分支”的评估意见：prediction summary 与 training compare 显式请求在目标未变化时优先复用 structured state，不再重复触发只读工具调用。
- 同批次已继续扩大到 prediction inspect 的显式目标请求：当 report/output 目标与当前 structured state 一致时，也直接走 cached result。
- 同批次已继续扩大到 prediction export / path-lists 的显式目标请求；同时 dataset duplicate 显式请求增加 target-aware cache gate，避免跨数据集误复用 structured state。

- 已继续落实“压缩 `chat()` 前置 code 拦截层”的评估意见：`prepare-only` 前置分支已并入 training-plan dialogue 入口，不再单独占据 mainline 顶层拦截。
- 已继续落实“减少 `intent_parsing.py` 消费端依赖”的评估意见：`agent_client.py` 已从按函数名导入收成模块级 `intent_parsing` 适配层，消费面继续收窄到本地 wrapper。

- 最新批次：extract / training-loop 这组只读 helper 路由已继续真实删减；`preview_extract_images`、`scan_videos`、`inspect_training_loop` 在目标未变化时优先复用 structured state。
- 最新批次：`agent_client.py` 已继续减少 `intent_parsing` 的消费端 wrapper，extract/realtime 这组 wrapper 已删除，消费面继续收窄。
- 最新批次：training-plan 对话前置层里的“无草案/无 pending 启动链”已抽成独立 bootstrap helper，`_try_handle_training_plan_dialogue` 顶层 pre-LLM 分支继续收薄。
- 最新批次：training 参数提取已统一收进 `_collect_requested_training_args` + `_apply_training_text_overrides`；训练/训练计划这组 `intent_parsing` wrapper 已批量删除，`agent_client.py` 中 `return intent_parsing.*` wrapper 现已降到个位数。


- 最新批次：training-plan dialogue 的 bootstrap 已从前置入口抽成 `_try_handle_training_plan_bootstrap`，training-plan frontdoor 再压一层。
- 最新批次：训练参数覆盖逻辑已统一进 `_apply_training_text_overrides`，`_collect_requested_training_args` 成为训练参数提取的共享入口。
- 最新批次：`agent_client.py` 中剩余的 `intent_parsing` wrapper 已继续成组删除，`return intent_parsing.*` wrapper 已清零。
- 最新批次：tool-result fallback 已改为“仅对受控的一组 training/realtime/remote 状态工具优先 grounded，其余优先 structured facts”，继续压缩 `grounded_reply_builder` 的主链影响面。
- 最新批次：测试基线已从硬编码 `C:\workspace\...` 路径收为仓内相对路径；`test_dataset_root_resolver.py` 也已改成跨平台断言，避免再向 C 盘写测试临时数据。
