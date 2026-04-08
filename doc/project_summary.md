# YoloStudio Agent 项目总结（2026-04-08）

> 本文档是项目交接文档，新会话直接读此文件即可续上。

---

## 一、项目目标

将 YoloStudio（PySide6 桌面 YOLO 工具）改造为 **LLM Agent 驱动的智能系统**。
用户在 Windows 终端用自然语言指挥服务器完成数据准备、模型训练等 YOLO 全流程。

---

## 二、已确认的技术决策

| 决策项 | 选择 |
|-------|------|
| 交互方式 | CLI Chat（MVP） |
| 运行位置 | Server 跑 MCP + Ollama，Windows 跑 Agent Client |
| 优先模块 | 数据准备 + 训练管理 |
| Claude/Cursor 兼容 | 预留，不做第一优先 |
| 传输层 | FastMCP + streamable-http（SSE 仅兼容） |
| 安全方案 | 服务绑定 127.0.0.1 + SSH Tunnel |
| LLM | Gemma4:e4b（GPU 0: RTX 3060） |
| YOLO 运行 | GPU 1: GTX TITAN X |

---

## 三、架构全景

```
   Windows 客户端                              服务器 192.168.0.163
  ┌─────────────────────┐    SSH Tunnel    ┌──────────────────────────┐
  │  cli.py             │   :8080/:11434   │  MCP Server (:8080)      │
  │  agent_client.py    │◄════════════════►│  FastMCP + streamable-http│
  │  LangGraph ReAct    │                  │  7 个 Tool               │
  │  langchain-mcp      │                  │  ├─ scan_dataset    ─┐   │
  │                     │                  │  ├─ split_dataset    │直接│
  │                     │                  │  ├─ validate_dataset │调用│
  │                     │                  │  ├─ augment_dataset ─┘   │
  │                     │                  │  ├─ start_training  ─┐   │
  │                     │                  │  ├─ check_status     │wrap│
  │                     │                  │  └─ stop_training   ─┘   │
  │  Ollama (:11434)◄═══╡                  │  Ollama (GPU 0: 3060)    │
  └─────────────────────┘                  │  YOLO   (GPU 1: TITAN X) │
                                           └──────────────────────────┘
```

**关键设计原则：**
- DataHandler 业务层（Mixin 方法）直接复用，无 Qt 依赖
- Train/Predict 需要 Service Wrapper（subprocess 替代 QProcess）
- 现有 `D:\yolodo2.0\core\` 源码零改动

---

## 四、项目文件结构

```
D:\yolodo2.0\agent_plan\               ← Git 仓库 (2 commits)
├── .gitignore
├── doc/
│   ├── yolostudio_agent_feasibility.md    # 可行性报告 (650行, 28KB)
│   ├── implementation_plan.md             # 实施计划 (290行)
│   ├── phase1_status.md                   # Phase 状态记录
│   └── project_summary.md                 # 本文档
│
├── agent/
│   ├── server/                            # 部署到服务器
│   │   ├── mcp_server.py                  # FastMCP 入口, 7个 Tool 注册
│   │   ├── tools/
│   │   │   ├── data_tools.py              # scan/split/validate/augment (157行)
│   │   │   └── train_tools.py             # start/status/stop (21行)
│   │   └── services/
│   │       ├── train_service.py           # subprocess 替代 QProcess (44行)
│   │       └── train_log_parser.py        # YOLO stdout 正则解析 (27行)
│   │
│   ├── client/                            # 运行在 Windows
│   │   ├── agent_client.py                # LangGraph Agent + HITL (181行)
│   │   └── cli.py                         # CLI Chat 入口 (49行)
│   │
│   └── tests/
│       └── test_gemma4_fc.py              # Function Calling 验证
│
└── deploy/                                # 服务器部署相关（自动生成）
```

**服务器部署位置：** `/home/kly/yolostudio_agent_proto`

---

## 五、Phase 完成度

### Phase 1: 基础设施 ✅ 全部完成

- [x] Ollama GPU 隔离（`CUDA_VISIBLE_DEVICES=0`，实机验证）
- [x] yolostudio-agent-server conda 环境 + 依赖
- [x] Gemma4 bind_tools 验证（add(3,5) 通过）
- [x] SSH Tunnel 连通性（8080 + 11434）
- [x] SSH 免密配置（ed25519，Host yolostudio）

### Phase 2: MCP Server ✅ 核心完成

- [x] FastMCP 启动（host/port 在构造函数中）
- [x] 7 个 Tool 注册并对齐真实 API
- [x] data_tools 4 工具：scan/split/validate/augment
- [x] train_tools 3 工具：start/status/stop
- [x] TrainService subprocess wrapper + 日志解析器
- [x] 服务器实机启动（PID 16397, :8080）
- [x] Windows SSH Tunnel 下 Tool 调用验证

### Phase 3: Agent Client ✅ 第一版完成

- [x] LangGraph ReAct Agent
- [x] Human-in-the-Loop（interrupt_before + 高/低风险区分）
- [x] CLI Chat（确认态、UTF-8、直接运行）
- [x] SYSTEM_PROMPT（中文、工具优先、不追问直接调用）
- [x] 端到端冒烟：低风险 check_status ✅ + 高风险 start_training 确认/取消 ✅

### Phase 4: 集成优化 ⏳ 未开始

- [ ] 完整场景测试（扫描→校验→拆分→训练→查状态 全流程）
- [ ] 参数校验/约束（device 等值范围）
- [ ] 对话上下文管理（长对话窗口溢出）
- [ ] 错误处理与异常恢复
- [ ] System Prompt 精调
- [ ] 使用文档

---

## 六、已知问题

| 问题 | 严重程度 | 位置 | 说明 |
|------|---------|------|------|
| **训练参数抽取不稳** | 🔴 | agent_client.py | LLM 有时把 `device=1` 提成 `device=0,1`。需要 prompt 约束或 Tool 端参数校验 |
| **每轮独立 thread_id** | 🟡 | agent_client.py L53 | 跨轮上下文靠 `self._messages` 手动传递，长对话会撑爆 LLM 窗口 |
| **split 只支持 train/val** | 🟡 | core API | 现有 `split_dataset()` 只做二分，test 集需要后续扩展 |
| **Tool 缺错误处理** | 🟡 | data_tools/train_tools | 异常直接抛出，没有友好的错误消息包装 |
| **MCP Server 没有重启脚本** | 🟢 | 运维 | 目前手动启动，没有 systemd 服务或 supervisor |

---

## 七、会话分工记录

### 当前会话（本对话）的工作

**角色：架构评审 + 文档质量把关**

1. **可行性报告编写与修正**（5 轮评审）
   - v1: 初始评估（LangChain @tool 方案）
   - v2: 转向 MCP Server 架构
   - v3: 用户审查 → 修正 5 处硬伤：
     - Qt 耦合程度被高估 → 精确到文件级依赖表
     - API 名/字段对不上 → 对齐 scan_dataset + ScanResult 真实字段
     - SSE 传输层偏旧 → FastMCP + streamable-http
     - 安全设计缺失 → 新增 5.6 安全与网络章节
     - 训练状态太轻 → 标注需开发日志解析器
   - v4: 用户审查 → 修正 3 处前后矛盾：
     - 传输层前后没统一 → 全文 streamable-http
     - 客户端示例和安全方案打架 → 统一 127.0.0.1 + SSH Tunnel
     - 架构描述自相矛盾 → 明确 DataHandler 直接复用 / Train+Predict 需 wrapper
   - v5: 用户审查 → 修正 API 名和 split 参数

2. **实施计划编写**
   - 4 项用户决策确认（CLI / Server+Client / 数据+训练 / 兼容预留）
   - 项目结构规划
   - Phase 1-4 任务清单 + 验收标准

3. **实机验证（本会话）**
   - SSH 到服务器验证：GPU 隔离、部署目录、conda 环境、MCP 端口
   - SSH config 权限修复（CodexSandboxUsers ACL 问题）
   - SSH 免密配置（ed25519 密钥 + Host yolostudio）
   - Git 初始化 + 提交管理

4. **代码审查**
   - Phase 2 代码审查（data_tools / train_tools / mcp_server 对齐真实 API 确认）
   - Phase 3 代码审查（agent_client.py / cli.py 架构和 HITL 逻辑确认）
   - 识别潜在问题（thread_id 策略、参数校验）

### 另一个会话的工作

**角色：实际编码 + 部署**

1. Phase 1: 服务器环境配置、依赖安装、Gemma4 验证
2. Phase 2: 编写全部 server 代码（mcp_server + tools + services）
3. Phase 2: 打包上传到服务器、启动 MCP Server、验证 Tool
4. Phase 3: 编写 agent_client.py（181行）和 cli.py（49行）
5. Phase 3: 端到端冒烟测试

---

## 八、给下一个会话的指引

### 如果要继续 Phase 4

1. 先读本文档了解全局
2. 检查服务器 MCP Server 是否还在跑：`ssh yolostudio "ss -tlnp | grep 8080"`
3. 如果没跑，到服务器启动：
   ```bash
   ssh yolostudio
   conda activate yolostudio-agent-server
   cd /home/kly/yolostudio_agent_proto
   python -m agent_plan.agent.server.mcp_server &
   ```
4. 在 Windows 开 SSH Tunnel：
   ```powershell
   ssh -L 8080:127.0.0.1:8080 -L 11434:127.0.0.1:11434 yolostudio
   ```
5. 跑 CLI：
   ```powershell
   cd D:\yolodo2.0
   python agent_plan\agent\client\cli.py
   ```

### 如果要修 bug

- 参数校验 → 改 `agent/server/tools/train_tools.py` 的 `start_training` 加 device 校验
- Prompt 约束 → 改 `agent/client/agent_client.py` 的 `SYSTEM_PROMPT`
- 上下文管理 → 改 `agent/client/agent_client.py` 的 thread_id 策略

### 关键文件速查

| 要做什么 | 看哪个文件 |
|---------|----------|
| 了解项目全局 | `doc/project_summary.md`（本文档） |
| 了解技术方案 | `doc/yolostudio_agent_feasibility.md` |
| 了解任务计划 | `doc/implementation_plan.md` |
| 改 MCP Tool | `agent/server/tools/data_tools.py` 或 `train_tools.py` |
| 改训练逻辑 | `agent/server/services/train_service.py` |
| 改 Agent 行为 | `agent/client/agent_client.py` |
| 改 CLI 交互 | `agent/client/cli.py` |
| 查真实 API | `D:\yolodo2.0\core\data_handler\_*.py` |

---

## 九、Git 状态

```
86e8e8c  init: Phase 1 完成 + Phase 2 骨架代码
297f71e  Phase 1+2 验收完成: GPU隔离/MCP启动/Tool验证/SSH免密
```

**未提交改动：**
- `agent/client/agent_client.py`（Phase 3 重写）
- `agent/client/cli.py`（Phase 3 重写）
- `doc/phase1_status.md`（状态更新）
- `doc/project_summary.md`（本文档，新增）

建议下次开始时先 commit 这些改动。
