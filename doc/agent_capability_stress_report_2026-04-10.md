# Agent 高强度能力测试报告（2026-04-10）

## 1. 测试范围

- Provider: `ollama + gemma4:e4b`、`deepseek + deepseek-chat`
- 场景类型：复杂多步训练、条件分支、上下文继承、取消后回忆、禁止启动、非标准目录容错
- 使用真实服务器、真实 MCP、真实数据集 `/data/test_dataset/`，并额外构造了非标准目录 `/data/agent_cap_tests/nonstandard_dataset/`

## 2. 总体结论

当前版本在**标准 YOLO 数据集 + 数据准备/训练主线**上已经很强，特别是 root path → prepare → readiness → train 这条链路已经稳定。
但在**非标准目录命名、状态污染、以及少量隐式参数推断**上仍有明确边界。

## 3. 逐项结果

| Provider | Case | 结论 | 说明 |
|---|---|---|---|
| ollama | root_train_chain | 通过 | 标准 root→prepare→train 链路已打通 |
| ollama | conditional_direct_train | 通过 | 已就绪数据集能跳过 split，直接训练 |
| ollama | followup_context_train | 部分通过 | 扫描后接“那就直接训练2轮”可成功，但模型会自行补默认模型，存在隐式推断 |
| ollama | status_branch | 部分通过 | 能正确只查状态不启动，但会把最近一次训练模型/数据写回新 session 状态 |
| ollama | nonstandard_dataset | 未通过 | 非标准目录下仍会进入 prepare 并执行 split，直到 generate_yaml 阶段才失败，失败点偏晚 |
| ollama | no_start_info | 通过 | 能遵守“不要启动训练”，并给出 YAML 与 auto 设备结论 |
| ollama | cancel_and_recall | 部分通过 | 取消后能回忆参数，但会把 device=auto 说成最近一次解析后的 1 |
| ollama | prepare_only_no_train | 通过 | 能只准备不训练 |
| deepseek | root_train_chain | 通过 | 标准复杂链路可稳定完成 |
| deepseek | nonstandard_dataset | 部分通过 | 不会误启动训练，但仍会先做 split，再因缺少类别信息失败 |
| deepseek | no_start_info | 通过 | 遵守不启动指令且输出更干净 |
| deepseek | cancel_and_recall | 通过 | 取消后参数回忆更贴近原始待确认参数（保留 auto） |

## 4. 关键发现

### 4.1 已经证明可用的能力
1. **复杂根目录提示词已打通**：像“数据在 /data/test_dataset/，按默认划分比例，然后用 yolov8n 模型训练 2 轮”这种提示词，现在能稳定进入 `prepare_dataset_for_training -> start_training` 两段确认链。
2. **readiness / no-start 场景表现稳定**：当用户要求“只检查、不要启动训练”时，当前版本能遵守约束。
3. **多轮上下文已经足够支持主线使用**：扫描后接“那就训练 2 轮”在 Gemma 路线上也能成功。
4. **DeepSeek 在说明性与取消后回忆上更稳**：尤其是保留 `device=auto` 这类原始参数时更准确。

### 4.2 暴露出的真实边界
1. **非标准目录命名仍是硬边界**：`pics/ann` 这种目录目前不会直接识别成标准结构。更重要的是，Agent 在这种情况下会先进入 `prepare_dataset_for_training` 并执行 `split_dataset`，直到 `generate_yaml` 缺少 classes 才失败。这说明失败点还不够早。
2. **状态查询会污染新 session 的训练上下文**：在 fresh session 中只做“如果有训练就停止，否则只告诉我没有训练”时，虽然行为正确，但 `SessionState.active_training` 会被服务器上最近一次训练残留信息填充。
3. **Gemma 仍有轻微隐式推断问题**：比如“扫描后直接训练 2 轮”时，会主动补 `yolov8n.pt`；取消后回忆参数时，会把 `device=auto` 回答成最近解析后的 `1`。这类问题不影响主链路，但说明它会把 tool 结果和推断混在一起。
4. **force_split 行为需要进一步明确**：当用户明确说“按默认划分比例再训练”时，Gemma 会把 `force_split=True` 传给 `prepare_dataset_for_training`，这在当前语义上是对的，但需要在 contract 里写得更清楚，避免后续被误认为“多余 split”。

## 5. 当前能力范围
### 适合当前版本直接处理的任务
- 标准 YOLO 目录（`images/` + `labels/`）
- 训练前检查 / readiness 判断
- root 路径到训练的两段式流程
- “不要启动训练，只告诉我结果” 这类约束型查询
- 训练取消后的基本参数回忆

### 当前版本仍有风险的任务
- 非标准目录命名（如 `pics/ann`）
- 完全省略模型名时，依赖模型自行补默认训练模型
- 依赖“当前无训练”这种状态判断的跨 session 精确记忆
- 需要非常精确保留原始 device 参数的解释型问答

## 6. 下一步建议（按优先级）
1. **把非标准目录失败点前移**：在 `prepare_dataset_for_training` 中，如果 resolver 给出 `unknown`，优先返回“请显式提供 img_dir/label_dir”，而不是先做 split。
2. **修正 training status 对 fresh session 的状态污染**：当 `check_training_status` 返回“当前没有训练任务”时，不应回填旧的 model/data_yaml 到新 session。
3. **收紧默认模型与 device 的解释规则**：避免 Gemma 在未明确要求时把“合理猜测”写成“确定参数”。
4. **继续扩大复杂提示词基线**：把本次 12 个 case 固化成长期回归集。

原始结果文件：`C:\workspace\yolodo2.0\agent_plan\agent\tests\test_agent_capability_range_output.json`