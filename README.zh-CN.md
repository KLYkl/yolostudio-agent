# YoloStudio Agent

[English](./README.md)

YoloStudio Agent 是 YoloStudio 对话式 YOLO 工作流背后的 agent / MCP 运行时仓库。
它把对话式 client、FastMCP tool server、规则型知识层，以及远端验证脚本整合在一起，用来把数据准备、训练、预测和结果追问收敛成可复用、可验证的 agent 流程。

**项目引用：**[KLYkl/YoloStudio](https://github.com/KLYkl/YoloStudio)

这个仓库是一个**基于 YoloStudio 独立开发**的 agent / MCP 工作区。

上面引用的 YoloStudio 仓库，是面向产品本体的桌面应用仓库：一个基于 PySide6 的 YOLO 数据集准备、模型训练和可视化推理桌面应用。
当前仓库沿用了这个产品领域背景和问题场景，但代码库本身是独立维护的，重点放在 agent runtime / MCP 工作流上。

这个仓库**不是**桌面 UI 主工程。
它聚焦的是 YoloStudio 的 agent 侧能力，用来：

- 把 YOLO 相关能力暴露成 MCP 工具
- 把自然语言请求路由成可执行工具链路
- 保存 grounded 会话状态，支持追问和延续性对话
- 对高风险或长任务提供确认 / HITL 边界
- 在本地和远端 GPU 主机上验证同一套工作流

## 和 YoloStudio 的关系

如果你是从 YoloStudio 仓库跳过来的，更准确的理解方式是：

- **来源 / 参考仓库**：[KLYkl/YoloStudio](https://github.com/KLYkl/YoloStudio)  
  重点是 PySide6 桌面产品本体和用户可见的应用模块。
- **当前仓库（`agent_plan`）**：独立维护的 agent / MCP 运行时工作区  
  重点是围绕同一类 YOLO 场景做对话编排、MCP 工具暴露、知识驱动追问，以及远端验证流程。

一句话说清：YoloStudio 是这个仓库的上游背景/来源参考，而当前仓库是基于它独立演进出来的 agent 向代码库，不是同仓配套子模块。

## 这个仓库为什么存在

Agent 层和桌面产品主工程的演进节奏并不完全一致，所以独立维护更合适：

- 可以独立迭代工具 schema、路由、记忆、确认策略
- 可以先验证 LLM-first 的流程，再决定怎么接入桌面主产品
- 可以对远端 roundtrip、回归矩阵、server mirror 做稳定维护
- 可以公开一个已脱敏、可审阅的 agent 工作区，而不把产品内部路径和敏感信息带出来

## 当前这个仓库能做什么

### 1. 数据集与数据准备工作流

当前 MCP server 已经覆盖常见 YOLO 数据处理动作，包括：

- 数据集扫描、校验、健康检查、重复图检测
- train/val 划分和 YAML 生成
- 标签修改、孤儿标签清理、空标签生成、缺失标签生成
- 标注格式转换和按类别归档
- 图片提取与视频抽帧
- 常用组合流 `prepare_dataset_for_training`

### 2. 训练工作流

当前运行时支持一条完整的训练导向对话主线：

- dataset readiness 与 execution readiness 分离
- 训练环境发现和 GPU 状态检查
- 训练 preflight
- 训练 start / status / stop
- run summarize / inspect / compare / best-run select
- 基于知识层的训练结果分析与下一步建议

### 3. 预测工作流

预测能力不只是单次图片推理，当前工具面包括：

- 图片 / 视频预测
- 异步图片预测会话
- 结果汇总、输出检查、报告导出
- 结果整理、路径列表导出
- camera / screen / RTSP 探测与 realtime prediction
- realtime 状态轮询与停止控制

### 4. Agent 运行时工作流

对话侧 client 当前负责：

- 意图解析与主线路由
- 基于结构化 tool result 的 grounded follow-up 回复
- 训练 / 预测 / 远端传输线程的会话状态保持
- 破坏性动作或长任务的 manual / auto confirmation 流程
- 最近轮次、事件、pending action 的本地持久化
- 在显式启用时接入 helper LLM 和 loop planner LLM

### 5. 远端验证与部署工作流

这个仓库还承载实际运维/验证用的脚本：

- 同步 `deploy/server_proto/` 下的受控 mirror
- 在远端 Linux 主机上启动 / 停止 / 检查 MCP server
- 运行远端 training / prediction / realtime RTSP / chaos 验证脚本
- 维护远端 profile 示例和桥接辅助脚本

## 架构概览

```text
User / CLI
  -> agent/client/
     - 基于 LangGraph 的对话运行时
     - session state、context retention、HITL、reply rendering
     - training / prediction / follow-up 请求服务
  -> MCP connection
  -> agent/server/mcp_server.py
     - FastMCP 工具注册
     - read-only / destructive 工具注解
     - 可选 loop planner 注入
  -> agent/server/tools/
     - data / extract / knowledge / predict / train / training_loop 工具层
  -> agent/server/services/
     - 数据、训练、预测、realtime、GPU 等实际服务逻辑
  -> knowledge/
     - 规则型训练知识和建议
  -> deploy/
     - mirror sync、远端 roundtrip、受控 server runtime
```

## 仓库结构

```text
agent/
  client/                 对话运行时、记忆、路由、HITL、CLI
  server/                 FastMCP server、tool surface、backend services
  tests/                  roundtrip、回归、chaos、契约验证
  AGENT.md                运行时 agent 行为契约
knowledge/                规则型知识库与 playbook
deploy/
  examples/               远端 profile 示例配置
  scripts/                mirror sync、远端 roundtrip、MCP 管理脚本
  server_proto/           远端受控 mirror + deploy-only 运行时依赖
    agent_plan/           远端机器使用的镜像包
    core/                 server 运行时依赖的 deploy-only 数据/运行时辅助模块
    utils/                server 运行时依赖的 deploy-only 通用工具模块
docs/                     当前设计说明与重构报告
memory/                   本地开发时的 session / event 持久化目录
```

## 当前代码体现的设计原则

这个仓库现在主要围绕几条比较稳定的约束在推进：

- **LLM-first，但必须 tool-grounded**：由模型结合工具事实做判断，而不是到处写死模板化路由。
- **readiness 分层**：数据集结构可训，不等于现在就可以开训；这是两类检查。
- **确认边界**：会改数据、会上传、会拉起长任务的动作，需要经过 pending-action 审批。
- **grounded follow-up**：对“刚才那个结果再解释一下”“继续说上一轮”的追问，优先复用最近结构化事实。
- **远端可复现**：远端验证依赖受控 mirror，而不是手工拼凑服务器目录。

如果你想理解当前方向背后的判断，优先看：

- `docs/agent-runtime-contract.md`
- `docs/agent-client-refactor-report.md`
- `docs/README.md`

## 快速开始

### 1. 安装依赖

client 侧依赖：

```bash
pip install -r agent/client/requirements_client.txt
```

server 侧依赖：

```bash
pip install -r agent/server/requirements_server.txt
```

### 2. 设置运行时导入路径

当前本地源码树启动时，需要把仓库根目录和 `deploy/server_proto` 一起放进 `PYTHONPATH`，因为 server 运行时仍然会复用 managed mirror 里的 `core/` 和 `utils/` 包。

PowerShell：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\deploy\server_proto"
```

bash：

```bash
export PYTHONPATH="$PWD:$PWD/deploy/server_proto"
```

### 3. 配置 LLM provider

默认路径是本地 Ollama。一个常见的本地配置示例：

PowerShell：

```powershell
$env:YOLOSTUDIO_LLM_PROVIDER = "ollama"
$env:YOLOSTUDIO_AGENT_MODEL = "gemma4:e4b"
$env:YOLOSTUDIO_OLLAMA_URL = "http://127.0.0.1:11434"
```

bash：

```bash
export YOLOSTUDIO_LLM_PROVIDER=ollama
export YOLOSTUDIO_AGENT_MODEL=gemma4:e4b
export YOLOSTUDIO_OLLAMA_URL=http://127.0.0.1:11434
```

client 也支持 `deepseek` 和 `openai_compatible` 两种 provider，按环境变量传入即可。

### 4. 启动 MCP server

```bash
python -m yolostudio_agent.agent.server.mcp_server
```

默认 MCP 地址是 `http://127.0.0.1:8080/mcp`。

如果是在远端 Linux 主机上，常用受控启动方式是：

```bash
APP_ROOT=/opt/yolostudio-agent CONDA_BIN=/opt/conda/bin/conda ENV_NAME=agent-server bash deploy/scripts/manage_mcp_server.sh restart
```

### 5. 启动交互式 CLI

```bash
python -m yolostudio_agent.agent.client.cli demo-session --confirm manual
```

常用参数：

- `--confirm manual|auto`
- `--provider ollama|deepseek|openai_compatible`
- `--model <model-name>`
- `--base-url <llm-base-url>`

### 6. 试几条代表性指令

以下示例比较贴合当前仓库的主线：

- `帮我检查这个数据集能不能直接训练：<dataset_root>`
- `把这个数据集整理成 YOLO 训练可用格式，必要时重建 data.yaml`
- `用 yolov8n.pt 在 <data_yaml> 上启动训练`
- `比较最近两次训练，告诉我哪个更好`
- `对这个图片目录跑一轮预测并总结结果`
- `测试这个 RTSP 地址是否可用，如果可用就启动实时预测`
- `解释一下最近这次训练为什么效果不好，并给我下一步建议`

## 关键环境变量

大多数情况下，先关心下面这些就够了：

| 变量 | 作用 |
| --- | --- |
| `YOLOSTUDIO_MCP_URL` | client 连接的 MCP 地址，默认 `http://127.0.0.1:8080/mcp` |
| `YOLOSTUDIO_CONFIRMATION_MODE` | `manual` 或 `auto`，控制 pending-action 审批行为 |
| `YOLOSTUDIO_LLM_PROVIDER` | 主模型 provider（`ollama`、`deepseek`、`openai_compatible`） |
| `YOLOSTUDIO_AGENT_MODEL` | 主对话模型 |
| `YOLOSTUDIO_LLM_BASE_URL` / `YOLOSTUDIO_LLM_API_KEY` | 兼容 provider 的 base URL 与凭证 |
| `YOLOSTUDIO_MEMORY_ROOT` | 会话 / 事件 / checkpoint 的持久化目录 |
| `YOLOSTUDIO_MAX_HISTORY_MESSAGES` | 历史消息保留窗口 |
| `YOLOSTUDIO_HELPER_LLM_*` | 辅助模型配置 |
| `YOLOSTUDIO_LOOP_LLM_*` | loop planner 模型配置 |
| `YOLOSTUDIO_ENABLE_LOOP_PLANNER` | 是否启用 host-side loop planner 注入 |
| `YOLOSTUDIO_TRAIN_DEVICE_POLICY` | 训练时的 GPU 分配策略 |
| `YOLOSTUDIO_GPU_BUSY_UTILIZATION_THRESHOLD_PCT` | GPU 忙碌阈值（利用率） |
| `YOLOSTUDIO_GPU_BUSY_COMPUTE_MEMORY_THRESHOLD_MB` | GPU 忙碌阈值（显存占用） |
| `YOLOSTUDIO_REMOTE_PROFILES_PATH` | 远端 profile JSON 配置路径 |

## 主要验证入口

这个仓库的重点不是做演示页面，而是验证工作流闭环。比较有代表性的验证切片包括：

### 元数据和 server surface 检查

```bash
python agent/tests/test_mcp_server_metadata.py
python agent/tests/test_server_smoke.py
```

### 训练 / 预测主线

```bash
python agent/tests/test_training_mainline_roundtrip.py
python agent/tests/test_prediction_route.py
python agent/tests/test_realtime_prediction_route.py
python agent/tests/test_remote_transfer_route.py
```

### 较长对话回归

```bash
python agent/tests/test_extreme_chat_regression.py
python agent/tests/test_training_loop_dialogue_matrix.py
```

### Chaos / 韧性验证

```bash
python agent/tests/test_agent_server_chaos_p0.py
python agent/tests/test_agent_server_chaos_p1_followup.py
python agent/tests/test_agent_server_chaos_p2_context.py
```

有些验证脚本依赖准备好的数据集、模型权重、可访问的 MCP server 或远端环境。把它当“一条命令就能过”的 smoke 之前，先看一下具体测试文件。

## 远端部署与 roundtrip 流程

这个仓库的部署路径建立在受控 mirror 上，而不是直接把开发目录 rsync 到线上。

典型流程：

1. 先刷新 managed mirror：

   ```bash
   python deploy/scripts/sync_server_proto.py
   ```

2. 查看远端 profile 示例：

   - `deploy/examples/remote_profiles.example.json`

3. 把 mirror 同步到目标机器，并运行远端 roundtrip 脚本，例如：

   - `deploy/scripts/run_training_remote_roundtrip.ps1`
   - `deploy/scripts/run_prediction_remote_roundtrip.ps1`
   - `deploy/scripts/run_realtime_rtsp_remote_roundtrip.ps1`
   - `deploy/scripts/run_training_agent_remote_followup_matrix.sh`

4. 在远端主机上用 `deploy/scripts/manage_mcp_server.sh` 重启或检查 MCP server。

## 当前范围与非目标

这个仓库明确只聚焦 YoloStudio 的 agent 侧。

当前范围：

- MCP 工具暴露
- agent runtime 行为
- 训练 / 预测 / 知识问答主线
- 远端验证与 roundtrip 可复现性

非目标：

- 桌面 GUI 本体
- 通用产品官网或营销页
- 把测试数据自动提升为正式知识
- 在公开仓库中保留本地路径、密钥、真实主机信息等敏感内容

## 知识层与数据卫生说明

`knowledge/` 下的知识源默认有分层策略：

- `official` 和 `workflow` 可以参与默认建议
- `case` 默认不自动启用
- `test` 默认禁止参与正式推荐链路

这个边界很重要：回归结果不能直接被当成“系统已经学会的最佳实践”。

## 建议优先阅读的文档

- `docs/agent-runtime-contract.md` —— 当前 runtime 接口与 pending-action 契约
- `docs/agent-client-refactor-report.md` —— 当前重构依据与方向；适合看设计判断，不适合当逐行代码导航
- `docs/README.md` —— `docs/` 下哪些文档是当前有效、历史参考或已归档
- `knowledge/README.md` —— 当前知识来源策略与防污染边界

## 发布 / 脱敏说明

公开版文档和脚本必须保持路径、主机、API key、环境细节的脱敏。
当前仓库中的示例值都应视为占位符，接入你自己的环境时需要自行替换。
