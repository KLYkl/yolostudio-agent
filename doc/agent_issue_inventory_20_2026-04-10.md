# Agent 20 类潜在问题逐项验证（2026-04-10）

说明：下面 20 条不是“都一定有问题”，而是基于当前架构筛出的 20 类典型风险，再逐条做动态验证或代码级验证。

| # | 问题类型 | 验证方式 | 结论 | 结果说明 |
|---|---|---|---|---|
| 1 | 标准 dataset root 是否还能扫错目录 | 动态测试：`scan_dataset(/home/kly/test_dataset/)` 与能力压测 root case | 未复现 | 已返回 33 张图，且解析为 `/images` + `/labels`。这类问题在标准 YOLO 结构下已修复。 |
| 2 | 已可直训的数据集会不会还被强制 split | 动态测试：Gemma `conditional_direct_train` | 未复现 | Gemma 在“如果可以直接训练就不要重新划分”场景下直接 `start_training`，未先 split。 |
| 3 | 扫描后再说“那就训练 2 轮”能否承接上下文 | 动态测试：Gemma `followup_context_train` | 未复现 | 扫描后接自然语言 follow-up 能直接训练，说明主线多轮承接已可用。 |
| 4 | 状态分支会不会误启动训练 | 动态测试：Gemma `status_branch` | 未复现 | “有训练就停，否则只告诉我没有训练”没有误触发 `start_training`。 |
| 5 | 非标准目录命名（pics/ann）是否能稳定识别 | 动态测试：Gemma/DeepSeek `nonstandard_dataset` | 确认存在 | 两条 provider 路线都未把 `pics/ann` 识别成标准 YOLO 结构。 |
| 6 | 非标准目录失败点是否过晚 | 动态测试：Gemma/DeepSeek `nonstandard_dataset` 查看 steps/transcript | 确认存在 | 当前会先进入 `prepare_dataset_for_training`，甚至先做 split，直到 `generate_yaml` / 缺类名时才失败，失败点偏晚。 |
| 7 | “不要启动训练，只告诉我结果”是否会被违背 | 动态测试：Gemma/DeepSeek `no_start_info` | 未复现 | 两条路线都遵守了“不启动训练”的约束。 |
| 8 | 取消后还能不能回忆刚才待确认参数 | 动态测试：Gemma/DeepSeek `cancel_and_recall` | 部分存在 | 两条路线都能回忆；但 Gemma 会把 `device=auto` 说成最近解析后的 `1`，DeepSeek 更准。 |
| 9 | Gemma 会不会擅自补默认训练模型 | 动态测试：Gemma `followup_context_train` | 确认存在 | 在用户未再次明确模型名时，Gemma 会自行补 `yolov8n.pt`。当前可用，但属于隐式推断。 |
| 10 | Gemma 会不会把 tool 结果解释成更确定的参数 | 动态测试：Gemma `cancel_and_recall` | 确认存在 | 取消后回忆参数时，Gemma 把原始 `device=auto` 解释成 `1`。 |
| 11 | fresh session 只查状态时会不会被最近训练污染 | 动态测试：Gemma `status_branch` 的 state snapshot | 确认存在 | 虽然当前无训练，但新 session 的 `active_training` 仍被最近一次训练的 model/data_yaml/device 填充。 |
| 12 | 不同 provider 对同一复杂指令的理解是否一致 | 动态测试：Gemma vs DeepSeek `root_train_chain` | 确认存在 | Gemma 按“默认划分比例”执行了 split；DeepSeek 最终直接沿用了现有 `data.yaml`，存在 provider 行为差异。 |
| 13 | 两段式高风险确认链是否稳定 | 动态测试：root_train_chain / complex prompt | 未复现 | 复杂训练链已能稳定出现 `prepare_dataset_for_training -> start_training` 两次确认。 |
| 14 | 只做准备不训练是否会误触发训练 | 动态测试：Gemma `prepare_only_no_train` | 未复现 | 该场景只执行 prepare，不会误启动训练。 |
| 15 | GPU 规则是否仍靠模型自由发挥 | 动态测试：`check_gpu_status`/`training_readiness` 远端实测 + 本地规则契约测试 | 未复现 | 工具层已显式返回 `device_policy / auto_device / next_actions`，主要规则已下沉到 service/tool。 |
| 16 | 训练工具返回是否仍然缺少统一语义 | 本地测试：`test_training_rules_contract.py` + 远端 `check_training_status/stop_training/start_training` | 未复现 | 训练相关工具已有 `summary / next_actions`，失败和空任务场景也有统一输出。 |
| 17 | MCP 重启后是否还能继续管理已启动训练 | 真实回归：启动训练 → 重启 MCP → fresh 进程调用 `check_training_status/stop_training` | 本轮已修 | 已新增持久化 run registry。真实服务器验证：MCP 重启后能显示 `reattached=true`，并可继续 `check_training_status` 与 `stop_training`。 |
| 18 | Agent 重启后是否还能继续 pending confirmation | 代码检查：使用 `MemorySaver()` + 进程内 checkpoint | 确认存在 | 当前 checkpoint 是 `MemorySaver()`，适合进程内恢复；Agent 进程重启后，图执行断点不具备持久恢复保障。 |
| 19 | DeepSeek tool schema 是否已启用 strict 约束 | 代码检查：`llm_factory.py` 仅接 `https://api.deepseek.com` + `ChatOpenAI`，未启用 beta strict | 确认存在 | 当前 provider 没启用 strict function schema，参数稳定性仍部分依赖 prompt 和 tool contract。 |
| 20 | 当前 HITL 机制是否完全是官方推荐的持久化做法 | 代码检查 + 官方文档对照：`interrupt_before=["tools"]` + `MemorySaver()` | 低优先级风险 | 当前做法能跑，但 LangGraph 官方更推荐用可持久 checkpoint 和更显式的人审中断模式做生产级 HITL。 |

## 结论汇总

- **当前仍存在的问题（高价值）**：8、9、10、12、18、19
- **已在后续主线中解决的问题**：5、6、11、17
- **低优先级架构风险**：20
- **本轮未复现/已较稳的能力**：1、2、3、4、7、13、14、15、16

## 当前最值得继续收口的 5 项

1. provider 间对“是否应 split”的行为差异
2. Gemma 对默认模型与 `device=auto` 的解释性漂移
3. DeepSeek strict tool schema / 更强参数约束
4. durable checkpoint / persistent HITL
5. CLI 恢复与故障提示再收口

原始压力测试结果：`D:\yolodo2.0\agent_plan\agent\tests\test_agent_capability_range_output.json`

## 后续进展（2026-04-10 晚间）

在后续主线收口中，以下问题已被补掉：
- #5 非标准目录命名识别不足：已通过 dataset root alias 支持覆盖常见 `pics/ann/imgs/annotations` 等目录名
- #6 非标准目录失败点过晚：`prepare_dataset_for_training` 现已在 `resolve_root` 提前阻断 truly unknown 结构
- #11 fresh session 训练状态污染：`check_training_status` 在 `running=false` 时只刷新 `last_status`
- #17 MCP 重启后无法继续管理已启动训练：已通过 run registry + reattach 修复

当前仍值得继续收口的主要是：
- #8 / #9 / #10：Gemma 对默认模型与 `device=auto` 的解释性漂移
- #12：provider 间对复杂训练意图的细微行为差异
- #18 / #20：durable checkpoint / persistent HITL
- #19：DeepSeek strict tool schema / 更强参数约束



## zyb 大数据测试补充发现（2026-04-10 夜）

新增发现的 3 个问题：
- `scan_dataset` 能看见大量缺失标签，但 `validate_dataset` / `training_readiness` 还不会把这类情况稳定提升为训练风险提示
- `prepare_dataset_for_training` 自动生成 YAML 时，当前会保留数字类 ID（`0/1/2/3`），不会优先利用 `labels/classes.txt` 写入真实类名
- Gemma 在脏数据解释和 prepare-only 场景下，仍可能生成超出工具事实范围的自然语言说明

## zyb 问题收口结果（2026-04-10 深夜）

此前 `zyb` 压测暴露出的两项主问题，本轮已完成主线修复：

1. **大量缺失标签没有被稳定提升为训练风险**
   - 已修复：`validate_dataset / training_readiness` 现在会显式返回
     - `missing_label_images`
     - `missing_label_ratio`
     - `risk_level`
     - `warnings`
2. **自动生成 YAML 时丢失真实类名**
   - 已修复：`generate_yaml / prepare_dataset_for_training` 现在会优先利用 `classes.txt`

对 `/home/kly/agent_cap_tests/zyb` 的远端真实验证结果：
- `class_name_source=classes_txt`
- `classes=[Excavator, bulldozer, piling_machine, two_wheeler]`
- `missing_label_images=5179`
- `missing_label_ratio=0.737`
- `risk_level=critical`
- `prepare_dataset_for_training(...).ready=true` 且 `summary` 会明确指出存在数据质量风险

## zyb 10 方法测试新增结论（2026-04-10 晚）

本轮 10 方法测试确认：

### 已证明较稳的范围
- large dirty dataset 下的 `scan / validate / readiness / prepare`
- `prepare -> start_training` 两段式主链路
- 真实训练 `start / status / stop`
- DeepSeek 在复杂训练链和风险解释上的整体稳定性

### 新确认的边界
1. **Gemma 的解释层仍不够 grounded**
   - 会把计数说虚
   - 会把类别名说错
   - 会把路径说成不存在的路径
2. **Gemma 的取消后参数回忆不够精确**
   - 能记住大意
   - 但具体 `data_yaml / epochs` 容易漂移
3. **执行链路已经强于解释链路**
   - 这说明当前主线的下一步重点，应是“解释层 grounded 化”和“待确认参数摘要结构化”
