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


- 最新批次：realtime 只读请求已开始走 structured-state 优先；`scan_cameras`、`scan_screens`、同目标 `test_rtsp_stream`，以及非 running 状态下的 `check_realtime_prediction_status` 均可直接复用 `last_realtime_status`。
- 最新批次：补了 realtime 缓存命中回归，确保这批 helper 删减是 facts-first 而不是局部最优；running 实时会话仍保持直连工具查询，避免把动态状态错误缓存化。

## 剩余 2 批的明确验收标准（2026-04-16）

### Batch 1：`grounded_reply_builder` 主链退出收口批
目标不是“继续压”，而是满足下面这些**可验收标准**：

1. **主链优先级标准**
   - 能由 renderer + structured facts 完成的主链回复，不再优先走 `grounded_reply_builder`。
   - `grounded_reply_builder` 只保留给：
     - planner / renderer 不可用
     - payload 缺少结构化 overview / action_candidates
     - 明确的兼容 fallback 路径

2. **代码位置标准**
   - 不新增新的主链业务逻辑到 `grounded_reply_builder.py`。
   - 若本批需要改 `grounded_reply_builder.py`，只允许：
     - 删除主链依赖
     - 收窄 fallback 触发条件
     - 修正 facts-first 显示

3. **只读 helper 删减标准**
   - 再删一批仍然安全的只读 helper direct-tool 路由。
   - 这批完成后，`agent_client.py` 里 `_complete_direct_tool_reply(` 的**调用点**必须继续下降，而不是只做 ownership 下沉。

4. **行为保护标准**
   - training / realtime / remote / prediction-management 这几条高频路径，不能因为缓存或 fallback 调整而把动态状态读成旧结果。
   - 对运行中状态（例如 running realtime / running training），仍以实时工具查询优先，不做错误缓存。

5. **Batch 1 验收通过条件**
   - 本批结束时：
     - `grounded_reply_builder` 不新增主链职责
     - `agent_client.py` 中 `_complete_direct_tool_reply(` 调用点较本批开始前继续下降
     - `planner_llm.ainvoke(` 保持不回升
   - 并通过一轮跨域回归：
     - pending / dialogue / roundtrip
     - training / training follow-up / loop
     - realtime / remote
     - prediction / prediction-management
     - grounded reply / adapter / facts

### Batch 2：最终验收批
目标也不是“再看看”，而是满足下面这些**收口标准**：

1. **测试基线一致性标准**
   - 本地与远端的关键测试基线不再存在“本地有、远端缺”的关键用例差异。
   - 测试路径不再写死到仓外绝对路径。

2. **跨域回归标准**
   - 跑完整跨域回归，至少覆盖：
     - training plan / pending / dialogue
     - training loop / roundtrip
     - prediction / prediction-management
     - dataset / extract
     - realtime / remote
     - grounded reply / tool adapter / tool result facts

3. **结构收口标准**
   - `agent_client.py` 不再出现新的前置 code-first 分支回流。
   - `intent_parsing.py` 不再新增语义路由职责。
   - `grounded_reply_builder.py` 保持 fallback 定位，不回流到主链职责。

4. **卫生项标准**
   - 文档留痕与测试基线同步完成。
   - 工作区干净，形成新的可审提交。

5. **Batch 2 验收通过条件**
   - 本地整批回归通过
   - 远端整批回归通过
   - 工作区 clean
   - 形成最终阶段性提交，供审查

## Batch 2 最终验收执行记录（2026-04-16）
- 最终验收口径：
  - 以 `agent_client.py` / `intent_parsing.py` / `grounded_reply_builder.py` 的当前结构统计为基线；
  - 以 training plan / pending / roundtrip / realtime / prediction / training follow-up / history / grounded reply / adapter / facts 作为跨域回归集合；
  - 远端验证环境统一为 `/opt/yolostudio-agent/agent_plan` + `yolostudio-agent-server`。
- 最终结构基线：
  - `agent_client.py`：8901 行
  - `intent_parsing.py`：558 行
  - `grounded_reply_builder.py`：1329 行
  - `planner_llm.ainvoke(`：2
  - `_complete_direct_tool_reply(`：27
  - `_complete_cached_tool_result_reply(`：31
  - `return intent_parsing.`：0
- 最终测试基线说明：
  - `test_training_plan_dialogue.py` 已补齐到远端，不再存在“本地有、远端缺”的差异。
  - 本地仍缺 `langchain_core`，因此 `test_tool_adapter.py` 与 `test_tool_result_facts.py` 在最终验收中继续以远端环境为准；其余主链回归本地可直接覆盖。


- Batch 1 最新批次：training/realtime/remote/history 这组高频只读查询已统一切到 `_complete_cached_or_direct_tool_reply`，不再在 helper 里散落“先 cached 再 direct_tool”的重复分支。
- Batch 1 最新批次：新增非 running 状态下的训练状态缓存门与环训练状态缓存门；running 训练/实时仍保持实时工具查询，避免把动态状态误缓存成旧结果。
- Batch 1 最新批次：`grounded_reply_builder` 的 fallback 触发面已进一步收窄——只有缺少 structured overview / action_candidates 的受控工具才优先 grounded，其余优先 structured facts。
- Batch 1 最新批次：`agent_client.py` 中 `_complete_direct_tool_reply(` 的文本出现次数已由上一批结束时的 45 降到 27，`planner_llm.ainvoke(` 仍保持 2，不回升。
## Batch 2 completed — final acceptance (2026-04-16)
- Final acceptance passed locally and remotely.
- `tool_adapter.py` now surfaces export/target paths in facts-first summaries (`export_path`, `export_dir`, `destination_dir`, `source_output_dir`, `source_report_path`), closing the last prediction-management acceptance gap.
- Current acceptance baseline passed on remote `yolostudio-agent-server` at `/opt/yolostudio-agent/agent_plan`:
  - `test_training_plan_dialogue.py`
  - `test_pending_confirmation_dialogue_route.py`
  - `test_training_loop_client_roundtrip.py`
  - `test_training_loop_dialogue_matrix.py`
  - `test_realtime_prediction_route.py`
  - `test_prediction_route.py`
  - `test_prediction_management_route.py`
  - `test_dataset_followup_route.py`
  - `test_extract_route.py`
  - `test_training_followup_route.py`
  - `test_training_history_followup_route.py`
  - `test_training_loop_history_followup_route.py`
  - `test_remote_transfer_followup_route.py`
  - `test_remote_pipeline_followup_route.py`
  - `test_grounded_tool_reply.py`
  - `test_tool_adapter.py`
  - `test_tool_result_facts.py`
- Batch 2 acceptance criteria are now satisfied; next step is final clean commit only.
