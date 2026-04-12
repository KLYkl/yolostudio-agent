# YoloStudio Agent Plan

[English](./README.md)

YoloStudio Agent Plan 是 YoloStudio 的 agent / MCP 工作区，用来把桌面端的 YoloStudio 工作流整理成一个可对话、可远端执行的训练与预测系统。

## 仓库包含内容

- **MCP 工具层**：覆盖数据集检查、数据准备、训练、预测、提取，以及训练知识解释
- **Agent 客户端**：负责意图解析、工具路由、确认流程、grounded reply 和会话状态
- **远端验证脚本**：用于训练与预测 roundtrip 验证
- **回归测试**：覆盖训练计划、训练主线 roundtrip、预测流程，以及部分远端 harness

## 仓库结构

```text
agent/                  Agent client、server、tools 和 tests
knowledge/              Phase 1 规则化知识库
deploy/                 远端验证脚本与 server prototype
doc/                    工程说明、playbook 与发布上下文
```

## 当前主线范围

本仓库当前主要聚焦：

1. **训练主线**：readiness -> prepare/preflight -> start -> status -> summarize -> analyze -> recommend
2. **预测主线**：图片/视频预测 -> 结果汇总 -> grounded replies
3. **远端 roundtrip**：通过远端 MCP server 验证同一套工作流

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
python -m agent_plan.agent.server.mcp_server
```

如果是在远端 Linux 主机上，也可以使用辅助脚本：

```bash
APP_ROOT=/opt/yolostudio-agent CONDA_BIN=/opt/conda/bin/conda ENV_NAME=agent-server bash deploy/scripts/manage_mcp_server.sh restart
```

### 3. 跑一组代表性测试

```bash
python agent/tests/test_training_mainline_roundtrip.py
python agent/tests/test_training_mainline_final_state_roundtrip.py
python agent/tests/test_training_status_route_phrases.py
```

在 Windows 客户端环境下，通常还会配合客户端虚拟环境运行长对话回归：

```powershell
.\agent\.venv\Scripts\python.exe .\agent\tests\test_extreme_chat_regression.py
```

## 远端验证脚本

`deploy/scripts/` 下面的脚本都已经做成适合公开仓库的示例版本，默认值是通用占位，不是你当前机器的真实环境。

通常需要覆盖的配置包括：

- 远端 SSH 目标或 host alias
- 远端 app root
- 远端数据集 / 模型路径
- 本地 stage / output 目录
- conda 环境名

## 文档与脚本中的路径约定

公开版文档与脚本使用的是通用示例值，例如：

- `C:\workspace\yolodo2.0`
- `/opt/yolostudio-agent`
- `/opt/conda`
- `/data/example_dataset`
- `/models/yolov8n.pt`
- `remote-agent`
- `203.0.113.10`

请根据你自己的环境替换这些路径和主机名。

## 说明

- 这个仓库有意把 **agent / MCP 工作区** 与主桌面产品代码分开。
- `doc/` 下仍保留了一部分工程说明，但已经对敏感本地路径和主机信息做过发布脱敏。
