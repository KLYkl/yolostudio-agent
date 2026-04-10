# YoloStudio Agent 当前项目进展（2026-04-09）

> 目的：给当前阶段做一次完整盘点，明确“已经完成什么 / 现在能做到什么 / 还剩什么问题 / 下一步该做什么”。

---

## 1. 项目目标回顾

当前 `D:\yolodo2.0\agent_plan` 的目标，不是直接改造主项目源码，而是先在 **独立原型层** 中完成一套可运行的 Agent 系统，围绕 YoloStudio 现有能力提供：

- 数据集扫描
- 数据集校验
- 数据集划分
- 数据增强
- YAML 生成
- 训练就绪判断
- 训练启动 / 状态查询 / 停止
- 自然语言多轮交互
- 高风险动作人工确认（HITL）
- 远端 MCP Server + 本地 Agent Client 架构
- 可替换的 LLM Provider（本地 Ollama / API 模型）

当前阶段的核心目标已经从“能不能做”进入到：

> **原型主链路是否已经打通，以及哪些部分还需要从“能跑”提升到“稳定可复用”。**

---

## 2. 当前总体状态结论

截至 2026-04-09，项目整体状态可以概括为：

> **工程化原型已经成型，主链路已打通，远端真实训练已验证；当前主要工作从“搭骨架”转入“规则收口、语义修正、稳定性增强”。**

更具体地说：

- ✅ **Agent 基础架构已成型**
- ✅ **MCP + Tool + CLI + HITL + Context Memory 已经联通**
- ✅ **DeepSeek 与 Ollama 都已接入同一套 Agent 架构**
- ✅ **服务器真实数据集与真实训练已经多轮验证**
- ✅ **多卡策略也已按真实 GPU 占用模型完成验证**
- ⚠️ **还存在 1 个明确主问题：数据集根目录语义歧义**
- ⚠️ **还存在一些“从可用到稳定”的收口项**

---

## 3. 当前架构形态

### 3.1 架构分层

当前原型已经形成了比较清晰的分层：

```text
Windows 本地
┌─────────────────────────────────────┐
│ Agent Client                        │
│ - cli.py                            │
│ - agent_client.py                   │
│ - context/memory/session state      │
│ - LangGraph ReAct + HITL            │
│ - LLM provider abstraction          │
└─────────────────────────────────────┘
                 │
                 │ SSH Tunnel
                 ▼
服务器 /home/kly/yolostudio_agent_proto
┌─────────────────────────────────────┐
│ MCP Server                          │
│ - mcp_server.py                     │
│ - data_tools.py                     │
│ - train_tools.py                    │
│ - train_service.py                  │
│ - gpu_utils.py                      │
│ - train_log_parser.py               │
└─────────────────────────────────────┘
                 │
                 ▼
        真实数据集 / YOLO / GPU / 训练进程
```

### 3.2 工作原则

当前原型遵循的关键工程原则是：

- **不直接修改主项目核心源码**
- 只在 `D:\yolodo2.0\agent_plan\...` 下开发 Agent 原型
- 对 `core` 做逻辑复用和外围封装，而不是侵入式改造
- 优先打通真实链路，再做更高维抽象

---

## 4. 已完成的核心能力

## 4.1 MCP Server 与工具层

当前 MCP Server 已经可以稳定运行在服务器上，监听：

- `127.0.0.1:8080`

当前已经注册并实测通过的 Tool 有 10 个：

1. `scan_dataset`
2. `split_dataset`
3. `validate_dataset`
4. `augment_dataset`
5. `generate_yaml`
6. `training_readiness`
7. `start_training`
8. `check_training_status`
9. `stop_training`
10. `check_gpu_status`

### 这些 Tool 已经达到的水平

不是“只有函数能调”，而是已经具备：

- 基本统一的 `ok / error / summary / next_actions` 结构
- 结构化返回，便于 Agent 复用
- 训练类工具已有前置校验与更清晰的返回字段
- 数据类工具已有摘要、候选 YAML、问题样例等信息

---

## 4.2 训练服务（TrainService）

`train_service.py` 目前已经不再是简单的命令封装，而是具备了以下能力：

- 使用 `subprocess.Popen` 替代 Qt 训练链路
- 自动搜索可用的 `yolo` 可执行文件
- 启动前校验：
  - `model`
  - `data_yaml`
  - `epochs`
- 运行中状态查询
- terminate → wait → kill 的停止兜底
- 训练设备策略解析
- 返回结构化训练信息：
  - `device`
  - `pid`
  - `log_file`
  - `command`
  - `resolved_args`
  - `device_policy`

### 当前已验证的训练模式

- 单卡自动选择：`single_idle_gpu`
- 全空闲卡自动多卡：`all_idle_gpus`
- 显式设备校验
- busy GPU 拒绝
- manual-only 策略要求显式 device

---

## 4.3 GPU 分配策略

这是最近一轮的重要升级。

项目已经从早期“假设某张卡保留给某模型”的思路，演进为：

> **基于真实 GPU 占用状态 + 可配置策略，动态解析训练设备。**

### 当前策略

位于：
- `agent/server/services/gpu_utils.py`

支持：

- `single_idle_gpu`
- `all_idle_gpus`
- `manual_only`

### 当前行为

- 自动查询 `nvidia-smi`
- 获取每张 GPU：
  - index
  - uuid
  - free_mb
  - busy
- 再按策略决定：
  - 选单张空闲卡
  - 选全部空闲卡
  - 或要求手动指定

### 已完成的实机验证

- 服务器 `single_idle_gpu` 下：`auto -> 1`
- 服务器 `all_idle_gpus` 下：`auto -> 0,1`
- 使用 `device=0,1` 的真实训练 2 epochs 已成功跑完

这说明：

> **系统已经不再被“本地 Ollama 占卡”的旧假设绑死，已经适配 API provider 场景和多卡空闲场景。**

---

## 4.4 Context / Memory 系统

这是 Agent 原型从“能聊”升级到“像系统”的关键。

### 已完成组件

- `session_state.py`
- `memory_store.py`
- `context_builder.py`
- `tool_result_parser.py`
- `event_retriever.py`

### 当前已经具备的上下文能力

#### 结构化状态持久化
会存储：

- 当前数据集：
  - `img_dir`
  - `label_dir`
  - `data_yaml`
  - `last_scan`
  - `last_validate`
  - `last_split`
- 当前训练：
  - `running`
  - `model`
  - `data_yaml`
  - `device`
  - `pid`
  - `log_file`
  - `last_status`
- 待确认操作：
  - `tool_name`
  - `tool_args`
  - `thread_id`

#### 事件日志
存放到：
- `agent_plan/memory/events/<session>.jsonl`

#### 会话状态
存放到：
- `agent_plan/memory/sessions/<session>.json`

#### 长对话控制
- 最近历史裁剪
- 事件摘要回灌 prompt
- 工具结果写回结构化状态
- 支持跨实例恢复

### 已验证结论

- 多轮上下文可延续
- Agent 重建后能恢复当前会话状态
- HITL 的待确认状态能持久化
- 长对话不再完全依赖原始 message history

---

## 4.5 HITL（高风险人工确认）

高风险 Tool 当前包括：

- `start_training`
- `split_dataset`
- `augment_dataset`

### 现在的行为

当 Agent 识别出高风险工具调用时：

- 不会直接执行
- 会先返回 `needs_confirmation`
- 客户端拦截并要求 `y/n`
- 再决定：
  - 执行
  - 取消

### 已验证情况

- 训练请求命中确认态：✅
- 取消路径：✅
- 确认执行路径：✅
- DeepSeek provider 下 HITL：✅

---

## 4.6 LLM Provider 抽象

这是当前阶段最重要的高维抽象之一。

### 已完成
新增：
- `agent/client/llm_factory.py`
- `agent/client/tool_adapter.py`

### 当前支持 Provider

- `ollama`
- `deepseek`
- `openai_compatible`

### 当前意义

系统已经从：

- 绑定 `ChatOllama + gemma4:e4b`

变成：

- 依赖统一 LLM provider 接口的可替换系统

### 已验证情况

#### Ollama
- 原链路仍可用：✅

#### DeepSeek
- API 可用：✅
- Tool 调用链可用：✅
- Agent 多轮 + HITL：✅

#### MiniMax
- 代理 OpenAI-compatible API 可通
- 但当前未作为优先 provider 深入打磨

---

## 5. 真实数据与真实训练验证情况

## 5.1 服务器现有数据集

当前服务器上已确认存在这些数据集：

### 1. `/home/kly/dataset_1ch`
- `images/train`
- `images/val`
- `labels/train`
- `labels/val`
- YAML：`/home/kly/dataset_1ch.yaml`

### 2. `/home/kly/test_dataset`
- `images`
- `labels`
- `images_split/...`
- `train_augmented/...`
- YAML：`/home/kly/test_dataset/data.yaml`

### 3. `/home/kly/test_dataset_split_for_yaml`
- `images/train`
- `images/val`
- `labels/train`
- `labels/val`
- YAML：`/home/kly/test_dataset_split_for_yaml/data.yaml`

---

## 5.2 已做的真实训练验证

### 训练验证矩阵

已经实际跑过：

1. `test_dataset` + `yolov8n.pt` + 4 epochs
   - **完整跑完**
   - return_code=0

2. `test_dataset` + `yolov8n.pt` + 6 epochs
   - **启动成功**
   - 中途查询成功
   - stop 成功
   - return_code=-15

3. Agent + DeepSeek 发起训练 + 3 epochs
   - scan → readiness → gpu → start_training(HITL) → confirm → check_status
   - **完整跑完**

4. 远端 `all_idle_gpus` 策略 + 2 epochs
   - `device=0,1`
   - **完整跑完**

### 结论

> **训练主链路已经不是“能启动”，而是“多次真实启动 / 查询 / 停止 / 完成”都验证过。**

---

## 6. 本轮新发现与新修复

## 6.1 已修复：latest_metrics 一直为 null

### 问题
`check_training_status()` 返回的：
- `latest_metrics` 长期为 `null`

### 根因
- YOLO 实际日志里包含 ANSI 转义
- 原正则太脆弱
- 还可能误匹配模型结构表等无关行

### 修复
修改：
- `agent/server/services/train_log_parser.py`

现在已支持：
- 去 ANSI 转义
- 只匹配真正 epoch 行
- 提取：
  - epoch
  - total_epochs
  - gpu_mem
  - box_loss
  - cls_loss
  - dfl_loss

### 当前状态
- 对已完成日志解析：✅
- 对远端真实日志验证：✅

### 仍需注意
在训练非常早期（例如刚启动 10 秒左右）有时仍可能拿不到 metrics，原因是：
- 日志里当时还没落到完整 epoch 行

这是正常时序问题，不再是 parser 逻辑 bug。

---

## 7. 当前仍然明确存在的问题

## 7.1 最高优先级问题：scan_dataset 根目录语义歧义

这是这轮综合验证里最清晰、最稳定复现的问题。

### 现象
如果用户说：
- “扫描 `/home/kly/test_dataset`”

Agent 当前会直接把：
- `img_dir=/home/kly/test_dataset`

传给 `scan_dataset`

结果会递归扫到：
- `images`
- `images_split`
- `train_augmented`
等子目录

导致结果：
- 根目录扫描：**78 张**
- 显式 images/labels 扫描：**33 张**

### 影响
这会造成：
- Agent 对数据集规模判断失真
- readiness 之前的理解偏差
- 用户普通说法和工具真实语义不一致

### 当前结论
> 这是现在项目最该优先修的主问题。

---

## 7.2 非标准数据集命名的容错仍有限

当前 dataset root resolver 已能稳定识别标准 YOLO 结构，例如 `images/`、`labels/`、train/val split 以及常见 `data.yaml` 命名；但如果用户目录命名不规范（如 `pics/`、`ann/` 等），当前更可能返回 `unknown` 或 `flat`，并提示用户显式提供 `img_dir` / `label_dir`。这比过去“静默扫错目录”更安全，但仍不够智能。后续应补：目录别名扩展、候选目录评分、以及 Agent 在 resolver 失败后的恢复式引导。

## 7.2 规则层还没有完全收口

虽然主链路已经通，但在以下方面还可以继续增强：

- Agent 对 GPU/device 规则的解释仍可能有自由发挥空间
- Tool description / prompt / service policy 三者还没完全统一
- 组合工具（如 `training_readiness`）还没成为 Agent 的默认强约束入口

这不是阻塞问题，但已经进入“稳定性优化”阶段。

---

## 7.3 文档、测试产物、仓库清洁度仍可继续整理

当前 `agent_plan` 仓库里仍有一些测试输出文件未处理，例如：

- `test_deepseek_long_context_smoke_output.json`
- `test_comprehensive_validation_output.json`

这些不影响主线，但后续应：
- 进 `.gitignore`
- 或转到专门的 `artifacts/` / `reports/` 路径

---

## 8. 当前已经达到什么程度

如果用阶段来定义，当前我会这样判断：

### 已经完成的阶段
#### 阶段 A：可行性确认
- 已完成

#### 阶段 B：原型骨架搭建
- 已完成

#### 阶段 C：主链路打通
- 已完成

#### 阶段 D：真实环境验证
- 已完成

### 当前正在进入的阶段
#### 阶段 E：稳定性与规则收口
- 正在进行中

也就是说，现在项目已经不是：
- “概念验证”
- “玩具 demo”

而是：

> **一个已经打通真实链路、具备继续产品化基础的工程原型。**

---

## 9. Git 与交付状态

### 已形成的关键提交

近期关键提交包括：

- `3de7d06` — structured context memory and event retrieval
- `1925381` — training / dataset tool outputs improvement
- `2ded8bc` — yaml generation and training readiness tools
- `76bc3f7` — provider abstraction and adaptive gpu allocation
- `5088a57` — docs and deployment notes update
- `afab4c1` — comprehensive validation + log parser fix

### 当前含义
说明当前原型不是零散修改，而是已经形成了：

- 连续演进记录
- 阶段性交付单元
- 可回溯的工程脉络

---

## 10. 建议的下一步主线

基于当前验证结果，下一步不建议继续盲目扩功能，建议进入：

## 10.1 第一优先：修正 scan_dataset 根目录语义

目标：
- 当用户给 dataset 根目录时，Agent 或 Tool 能更合理地推导：
  - `images/`
  - `labels/`

可选方向：
- Agent prompt / tool description 收紧
- Tool 层新增 root-path helper
- 或 scan 增加 dataset-root 模式

这是最值得优先解决的问题。

## 10.2 第二优先：规则收口

重点：
- 将 GPU 策略说明进一步写进 Tool contract
- 明确 `training_readiness` / `check_gpu_status` / `start_training` 三者之间的规则边界
- 降低 Agent 的自由发挥空间

## 10.3 第三优先：回归基线化

现在已经有：
- DeepSeek 长上下文测试
- 综合验证脚本
- 真实训练验证

下一步应把它们整理成：
- 稳定的回归基线
- 明确哪些脚本属于 smoke，哪些属于 full validation

---

## 11. 一句话结论

> **截至 2026-04-09，YoloStudio Agent 项目已经完成了从“可行性论证”到“真实训练主链路打通”的关键跨越。当前最重要的工作不再是补骨架，而是收口规则，尤其优先解决 scan_dataset 的根目录语义歧义。**

---

## 12. 2026-04-10 新增能力范围验证记录

本轮已完成 20 类潜在问题的逐项验证，结果已固化到：
- `D:\yolodo2.0gent_plan\docgent_issue_inventory_20_2026-04-10.md`
- `D:\yolodo2.0gent_plan\docgent_capability_stress_report_2026-04-10.md`
- `D:\yolodo2.0gent_plangent	ests	est_agent_capability_range_output.json`

### 新确认的问题边界
- 非标准目录命名（`pics/ann`）仍不能稳定识别为标准 YOLO 数据集
- 非标准目录当前失败点偏晚，会先进入 prepare / split，再到 YAML 阶段失败
- fresh session 下只查训练状态时，`active_training` 会被最近训练信息污染
- Gemma 在默认模型、`device=auto` 的解释上仍有轻微“合理猜测过头”现象
- provider 之间对复杂训练意图（尤其是否需要 split）仍可能出现不同决策

### 新确认的稳定能力
- 标准 root 路径输入已不再误扫到 78 张，当前能稳定解析为 33 张
- root → prepare → start_training 的两段式确认链稳定
- “只检查、不训练” 场景稳定
- 取消后回忆最近待确认参数已基本可用（DeepSeek 更稳）

---

## 13. 2026-04-10 主线鲁棒性收口进展

本轮继续沿主线推进，没有扩新业务能力，重点是把“数据集根目录 → prepare → train”这条链做得更稳。

### 已完成

#### 1) 非标准目录容错增强
- `dataset_root.py` 现在支持更多别名：
  - 图片目录：`pics / imgs / imageset / pictures / jpegimages`
  - 标签目录：`ann / anns / annotations / label / txt_labels`
- 这让非标准但常见的数据集命名不再一律落到 `unknown`。

#### 2) 失败点前移
- `prepare_dataset_for_training` 现在会在 `resolve_root` 阶段就拦住真正未知结构：
  - `unknown`
  - `images_only`
  - `flat`
- 不再像之前那样继续进入 scan / split / generate_yaml 才失败。

#### 3) fresh session 训练状态纯净化
- `check_training_status` 现在当 `running=false` 时，只刷新 `last_status`；
- 不再把服务端最近一次训练的 `model / data_yaml / device` 回写进 fresh session 的 `active_training`。

### 已验证

#### 本地验证
- `py_compile` ✅
- `test_dataset_root_resolver.py` ✅
- `test_prepare_dataset_flow.py` ✅
- `test_training_state_purity.py` ✅

#### 远端真实验证
- `resolve_dataset_root('/home/kly/agent_cap_tests/nonstandard_dataset')` ✅
  - 现可识别 `pics/ann`
- `prepare_dataset_for_training('/home/kly/agent_cap_tests/nonstandard_dataset')` ✅
  - 现可完整走到 ready=true
- `prepare_dataset_for_training('/home/kly/agent_cap_tests/unknown_dataset')` ✅
  - 现会在 `blocked_at=resolve_root` 提前失败
- Gemma 复杂提示词回归（fresh session）✅
  - 重新回到两段式确认链：
    1. `prepare_dataset_for_training`
    2. `start_training`

### 当前主线判断

当前主线已经从：
- “标准目录能跑”
推进到：
- “常见非标准目录也能处理，真正未知目录能尽早阻断，session 状态不再轻易被旧训练污染”

这一步覆盖了前一轮高强度测试里最值钱的 3 类问题：
1. 非标准目录命名识别不足
2. 非标准目录失败点过晚
3. fresh session 状态污染

### 下一步主线

下一步不建议扩功能，仍然继续收口：
1. Gemma / DeepSeek 对复杂训练意图的解释一致性
2. 训练参数“默认推断”与“用户明确指定”的措辞边界
3. CLI / 文档里的恢复指引与失败可观测性

---

## 14. 2026-04-10 复杂训练意图一致性收口

本轮进一步收口了主线里最容易让 provider 分叉的部分：

### 已完成

#### 1) 参数来源显式化
- `training_readiness` 现在会返回：
  - `data_yaml_source`
  - `recommended_start_training_args`
- `prepare_dataset_for_training` 现在会返回：
  - `force_split_applied`
  - `split_reason`
  - `data_yaml_source`
  - `recommended_start_training_args`
- `start_training` / `train_service` 现在会返回：
  - `requested_device`
  - `argument_sources`

#### 2) Prompt / contract 收口
- 如果用户明确表达“按默认比例划分 / 先划分再训练”，应传 `force_split=true`
- 如果工具已经给出 `args_hint` / `recommended_start_training_args`，后续优先原样复用
- 回答参数时要区分：
  - 用户明确指定
  - 工具检测/生成
  - auto 解析

#### 3) 主线控制器兜底
- 在 `prepare_dataset_for_training` 已确认成功、用户原始意图包含训练、但模型自己没有继续发出 `start_training` tool call 时，客户端会按当前会话状态自动合成下一步 `start_training` 确认请求。
- 这个逻辑只服务主线，不是通用“替模型做规划”。

### 已验证

#### Gemma 路线
复杂提示词：
- `数据在 /home/kly/test_dataset/，按默认划分比例，然后用yolov8n模型进行训练`

结果：
1. `prepare_dataset_for_training(force_split=true)` 确认
2. `start_training(data_yaml=..., model=yolov8n.pt)` 确认

#### DeepSeek 路线
同一提示词回归通过：
1. `prepare_dataset_for_training(force_split=true)`
2. `start_training(...)`

### 当前判断

经过这一轮，当前主线已经不只是“能跑”，而是：
- 标准路径稳定
- 常见非标准目录可处理
- truly unknown 结构可提前阻断
- 复杂训练意图在双 provider 下都能回到一致的两段式流程

也就是说：
> **已经接近可以开始谨慎增加功能的节点。**

如果继续稳一轮，我建议最后再补：
1. 训练 run registry / MCP 重启后的训练接管
2. CLI 恢复与故障提示再收口

## 15. 2026-04-10 训练 run registry / MCP 重启接管

本轮继续沿主线推进，没有扩新业务，而是补上了主线最后一个明显的系统缺口：

### 已完成

#### 1) 训练 run registry
- `train_service.py` 现在会把当前训练任务持久化到：
  - `runs/active_train_job.json`
- 训练结束、停止或进程消失后，会落最近一次运行信息到：
  - `runs/last_train_job.json`
- 持久化字段包括：
  - `pid`
  - `log_file`
  - `started_at`
  - `requested_device / device`
  - `command`
  - `resolved_args`
  - `argument_sources`

#### 2) MCP 重启后的训练接管
- fresh `TrainService()` 现在会在 `status()/stop()/start()` 前自动读取 active registry。
- 如果发现注册表里的 pid 仍在运行：
  - 会把当前任务重新装载到 runtime 中
  - `check_training_status` 会返回 `reattached=true`
  - `summary` 会明确提示“已从注册表接管”
- 如果发现 pid 已不在运行：
  - 会自动把 active registry 归档成 last run
  - 避免 registry 残留导致假阳性

#### 3) MCP 重启后仍可 stop
- 如果 MCP 重启后 `_process` 句柄已丢失，`stop()` 仍可基于注册表中的 pid 发送终止信号。
- 这意味着主线现在已经具备：
  - 启动训练
  - 重启 MCP
  - 再次查询训练
  - 再次停止训练
  的完整接管能力。

### 已验证

#### 本地验证
- `py_compile` ✅
- `agent/tests/test_train_run_registry.py` ✅
  - 模拟：写入 active registry → 新建 `TrainService()` → `status()` 重新接管 → `stop()` 成功结束 → last registry 正确落盘

#### 远端真实验证
- 已同步最新 `train_service.py` 到 `/home/kly/yolostudio_agent_proto`
- 真实服务器验证链路：
  1. 启动真实训练（`epochs=50`）
  2. 重启 MCP Server
  3. fresh 进程调用 `check_training_status`
  4. 观察到：
     - `running=true`
     - `reattached=true`
     - `summary=训练进行中 ... 已从注册表接管`
  5. fresh 进程调用 `stop_training`
  6. 成功停止，返回 `return_code=-15`

### 当前主线判断

到这一步，主线已经补上了“训练进行中时 MCP 重启会丢失控制权”的明显缺口。

也就是说当前主线已经具备：
- root → prepare → train
- 双 provider 复杂训练意图收口
- 非标准目录容错与早失败
- fresh session 状态纯净化
- 训练 run registry / MCP 重启接管

### 剩余主线重点

如果继续保持“先稳再扩”，当前主线剩余最值得做的只剩：
1. durable checkpoint / persistent HITL（替换 `MemorySaver()`）
2. provider 在复杂训练意图上的剩余解释差异再收一轮
3. CLI 恢复与故障提示收口
