# YoloStudio Agent

[English](./README.md)

YoloStudio Agent 是 YoloStudio 的 agent / MCP 工作区，用来承载可对话的训练、预测、知识解释与远端验证能力。
这个仓库把 MCP server、对话式 agent client、规则化知识库，以及远端验证脚本整理在一起，用于把桌面侧的 YOLO 操作流程收敛成可复用、可验证的 agent 工作流。

## 这个仓库是做什么的

这个仓库聚焦的是 **YoloStudio 的 agent 侧能力**：

- 提供数据集、训练、预测、提取、知识查询等 MCP 工具
- 把自然语言请求路由成可执行的工具链路
- 保存 grounded 状态，支持追问、确认、总结与建议
- 用本地测试和远端 roundtrip 验证同一套工作流

它有意和桌面主产品代码分开，方便 agent / MCP 层独立演进、测试和发布。

## 能力概览

### 训练工作流

- 数据集 readiness / training preflight
- 数据集准备 / YAML 自动修复与重建
- 训练 start / stop / status / summarize / inspect
- 训练结果分析与下一步建议
- 训练产物回传与远端 follow-up 验证

### 预测工作流

- 图片 / 视频预测
- 预测结果汇总与 grounded follow-up
- 预测结果管理与导出辅助
- realtime / RTSP 预测工作流

### Agent 工作流

- 意图解析与工具路由
- 确认流程与 guardrail 处理
- 会话状态与 grounded replies
- 基于规则知识库的训练解释与建议

## 仓库结构

```text
agent/                      事实上的源码目录：agent client、server、tools、tests
knowledge/                  Agent 使用的规则化知识库
deploy/scripts/             远端验证、roundtrip、bridge 脚本
deploy/server_proto/        远端部署路径使用的镜像原型包
docs/                       公开保留的工作流说明与测试计划
```

## 当前公开主线范围

当前主线主要围绕三条路径：

1. **训练主线**  
   readiness -> prepare/preflight -> start -> status -> summarize -> analyze -> recommend
2. **预测主线**  
   图片/视频预测 -> 结果汇总 -> 结果追问 -> grounded replies
3. **远端与实时验证**  
   远端 roundtrip、训练产物回传，以及 RTSP / realtime prediction 验证

## 快速开始

### 1. 安装依赖

客户端依赖：

```bash
pip install -r agent/client/requirements_client.txt
```

服务端依赖：

```bash
pip install -r agent/server/requirements_server.txt
```

### 2. 启动 MCP server

```bash
python -m yolostudio_agent.agent.server.mcp_server
```

如果是在远端 Linux 主机，也可以用辅助脚本：

```bash
APP_ROOT=/opt/yolostudio-agent CONDA_BIN=/opt/conda/bin/conda ENV_NAME=agent-server bash deploy/scripts/manage_mcp_server.sh restart
```

### 3. 跑一组代表性验证

```bash
python agent/tests/test_training_mainline_roundtrip.py
python agent/tests/test_prediction_route.py
python agent/tests/test_realtime_prediction_route.py
python agent/tests/test_remote_transfer_route.py
```

在 Windows 客户端环境下，通常还会配合本地虚拟环境运行更长的对话回归：

```powershell
.\agent\.venv\Scripts\python.exe .\agent\tests\test_extreme_chat_regression.py
```

## 远端验证脚本

`deploy/scripts/` 下的脚本都是适合公开仓库的示例版本，默认值是通用占位，需要你按自己的环境覆盖。

常见需要替换的配置包括：

- SSH host alias 或远端目标
- 远端 app root
- 远端数据集 / 模型路径
- 本地 stage / output 路径
- conda 环境名

代表性脚本包括：

- `deploy/scripts/run_training_remote_roundtrip.ps1`
- `deploy/scripts/run_realtime_rtsp_remote_roundtrip.ps1`
- `deploy/scripts/run_prediction_remote_roundtrip.ps1`
- `deploy/scripts/run_training_agent_remote_followup_matrix.sh`

## 路径约定

公开版文档与脚本使用的是通用示例值，例如：

- `C:\workspace\yolodo2.0`
- `/opt/yolostudio-agent`
- `/opt/conda`
- `/data/example_dataset`
- `/models/yolov8n.pt`
- `remote-agent`
- `203.0.113.10`

请根据你的真实环境替换这些路径和主机名。

## 说明

- `agent/` 是事实上的源码目录；`deploy/server_proto/` 是远端部署链路用到的镜像包。
- `docs/` 只保留适合公开仓库的说明与计划。
- 本地机器路径、主机名、密钥和敏感环境信息在发布前都应脱敏或替换为占位值。
