# YoloStudio Agent 项目总结（2026-04-08 更新）

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
| YOLO 运行 | 动态检测空闲 GPU（默认 auto） |

---

## 三、架构全景

```
   Windows 客户端                              服务器 192.168.0.163
  ┌─────────────────────┐    SSH Tunnel    ┌──────────────────────────┐
  │  cli.py             │   :8080/:11434   │  MCP Server (:8080)      │
  │  agent_client.py    │◄════════════════►│  FastMCP + streamable-http│
  │  LangGraph ReAct    │                  │  8 个 Tool               │
  │  langchain-mcp      │                  │  ├─ scan_dataset    ─┐   │
  │                     │                  │  ├─ split_dataset    │直接│
  │                     │                  │  ├─ validate_dataset │调用│
  │                     │                  │  ├─ augment_dataset ─┘   │
  │                     │                  │  ├─ start_training  ─┐   │
  │                     │                  │  ├─ check_status     │wrap│
  │                     │                  │  ├─ stop_training    │    │
  │                     │                  │  └─ check_gpu_status ─┘   │
  │  Ollama (:11434)◄═══╡                  │  Ollama (GPU 0: 3060)    │
  └─────────────────────┘                  │  YOLO   (auto GPU)       │
                                           └──────────────────────────┘
```

**关键设计原则：**
- DataHandler 业务层（Mixin 方法）直接复用，无 Qt 依赖
- Train/Predict 需要 Service Wrapper（subprocess 替代 QProcess）
- 训练 GPU 动态检测（查 compute 进程，不硬编码设备号）
- 现有 `D:\yolodo2.0\core\` 源码零改动

---

## 四、项目文件结构

```
D:\yolodo2.0\agent_plan\               ← Git 仓库 (4 commits)
├── .gitignore
├── doc/
│   ├── yolostudio_agent_feasibility.md    # 可行性报告 (650行, 28KB)
│   ├── implementation_plan.md             # 实施计划 (290行)
│   ├── phase1_status.md                   # Phase 状态记录
│   └── project_summary.md                 # 本文档
│
├── agent/
│   ├── server/                            # 部署到服务器
│   │   ├── mcp_server.py                  # FastMCP 入口, 8个 Tool 注册
│   │   ├── tools/
│   │   │   ├── data_tools.py              # scan/split/validate/augment (157行)
│   │   │   └── train_tools.py             # start/status/stop/gpu_status (27行)
│   │   └── services/
│   │       ├── train_service.py           # subprocess + 设备校验 (133行)
│   │       ├── gpu_utils.py               # GPU 动态检测（进程检查+UUID映射）
│   │       └── train_log_parser.py        # YOLO stdout 正则解析 (27行)
│   │
│   ├── client/                            # 运行在 Windows
│   │   ├── agent_client.py                # LangGraph Agent + HITL (192行)
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
- [x] 8 个 Tool 注册并对齐真实 API
- [x] data_tools 4 工具：scan/split/validate/augment
- [x] train_tools 4 工具：start/status/stop/gpu_status
- [x] TrainService subprocess wrapper + 日志解析器
- [x] **GPU 动态检测**：gpu_utils.py（查 compute 进程 + UUID→index 映射）
- [x] **设备校验**：device=auto 自动选卡，拒绝 cpu/多卡/busy卡
- [x] 服务器实机启动（PID 16397, :8080）
- [x] Windows SSH Tunnel 下 Tool 调用验证

### Phase 3: Agent Client ✅ 第一版完成

- [x] LangGraph ReAct Agent
- [x] Human-in-the-Loop（interrupt_before + 高/低风险区分）
- [x] CLI Chat（确认态、UTF-8、直接运行）
- [x] SYSTEM_PROMPT（中文、工具优先、不追问直接调用）
- [x] 对话上下文裁剪（`_trim_history` + `max_history_messages=12`）
- [x] 端到端冒烟：低风险 check_status ✅ + 高风险 start_training 确认/取消 ✅

### Phase 4: 集成优化 ⏳ 部分完成

- [x] **完整场景测试**：scan → validate → split → augment → start_training → check_status 全流程
- [x] **错误处理**：data_tools / train_tools 已加 try-except + _error_payload 统一包装
- [x] **train_service 前置校验**：epochs/data_yaml/yolo 命令检查
- [x] **train_service 进程停止兜底**：terminate → wait → kill
- [x] **服务器代码同步**：已上传并重启 MCP（PID 25915）
- [x] **冒烟测试脚本**：`tests/test_server_smoke.py`
- [ ] System Prompt 精调
- [ ] 使用文档
- [ ] MCP Server 标准启动/重启脚本（systemd/supervisor）

---

## 六、已知问题

| 问题 | 严重程度 | 位置 | 说明 |
|------|---------|------|------|
| ~~训练参数抽取不稳~~ | ~~🔴~~ | ~~train_service~~ | ✅ **已修**：device 默认 `auto`，gpu_utils 动态检测 |
| ~~上下文窗口溢出~~ | ~~🟡~~ | ~~agent_client.py~~ | ✅ **已修**：`_trim_history()` 保持 tool_call↔ToolMessage 配对 |
| ~~split mode 幻觉~~ | ~~🟡~~ | ~~data_tools~~ | ✅ **已修**：docstring 加了 `mode 可选值: copy/move/index` |
| ~~split 路径丢失~~ | ~~🟡~~ | ~~data_tools~~ | ✅ **已修**：返回值加了 `output_dir` 绝对路径 |
| ~~yolo 命令找不到~~ | ~~🔴~~ | ~~train_service~~ | ✅ **已修**：`_find_yolo_executable()` 自动搜索 conda 环境 |
| ~~CLI 崩溃 INVALID_CHAT_HISTORY~~ | ~~🔴~~ | ~~agent_client~~ | ✅ **已修**：`_trim_history` 保持配对完整性 |
| **latest_metrics 始终 null** | 🟡 | train_log_parser | 正则未匹配 YOLO 实际输出格式，需要检查 |
| **generate_yaml 未暴露** | 🟡 | data_tools | core 有 `generate_yaml` 方法但未注册为 MCP Tool |
| **MCP 重启后训练丢失** | 🟡 | train_service | 进程引用是内存级的，MCP 重启后无法恢复 |
| **split 只支持 train/val** | 🟡 | core API | 现有 `split_dataset()` 只做二分 |
| **MCP Server 没有重启脚本** | 🟢 | 运维 | 目前手动启动 |
| ~~模型权重损坏~~ | ~~🔴~~ | ~~服务器~~ | ✅ **已修**：`yolov8n.pt`(380K→6.3M) 和 `yolo26n.pt`(496K→5.3M) 均被损坏文件覆盖，从 `/home/kly/` 正确副本恢复 |
| **scan_dataset 递归扫描子目录** | 🟡 | data_tools / core | scan 传入 `test_dataset` 会递归扫描 images_split/train_augmented 等子目录，导致图片数虚高（33→78），需明确传 `img_dir` 和 `label_dir` |
| **Agent 不自动推导 img/label 目录** | 🟡 | System Prompt / Agent | 用户说「扫描 /home/kly/test_dataset」时，Agent 直接把整个目录传给 `img_dir`，未自动推导出 `images/` 和 `labels/` 子目录 |
| **Ollama 僵死占端口** | 🟢 | 服务器运维 | Ollama 进程(PID 15713)占着 11434 端口但 connection reset，需 kill 重启 |

---

## 七、会话分工记录

### 会话 1（a8261698）：架构评审 + GPU 方案设计

1. 可行性报告编写与修正（5 轮评审）
2. 实施计划编写（Phase 1-4 任务清单）
3. 实机验证（SSH/GPU/conda/MCP）
4. GPU 动态检测方案设计与实现

### 会话 2（编码会话）：Phase 1-3 实际编码 + 部署

1. Phase 1: 服务器环境配置、Gemma4 验证
2. Phase 2: 编写全部 server 代码 + 部署启动
3. Phase 3: 编写 agent_client.py + cli.py + 冒烟测试

### 会话 3（4e533b01）：Phase 4 闭环测试 + Bug 修复

**完成的工作：**

1. **测试数据准备**：33 张图 + 25 标签上传到服务器 `/home/kly/test_dataset/`
2. **8 个 MCP Tool 全部验证通过**：
   - scan_dataset ✅ | validate_dataset ✅ | split_dataset ✅ | augment_dataset ✅
   - start_training ✅ | check_training_status ✅ | stop_training ✅ | check_gpu_status ✅
3. **修复 4 个 Bug**：
   - split mode 幻觉（docstring 说明合法值）
   - split 返回值缺绝对路径（加 output_dir 字段）
   - yolo 命令找不到（`_find_yolo_executable` 搜索 conda 环境）
   - CLI 崩溃 INVALID_CHAT_HISTORY（`_trim_history` 配对保护）
4. **训练实测**：yolov8n.pt 成功启动到 GPU 1，自动选卡正确
5. **新增知识项**：硬件无关性原则、服务器连接规则

**教训：**
- SSH 连接管理差（频繁新建一次性连接导致卡死）→ 已写入知识项
- 应先建立 SSH 连接再操作服务器

### 会话 4（882074fa，本次）：AI Agent 端到端训练测试

**目标：** 在真实环境中测试 AI Agent 完整训练流程

**完成的工作：**

1. **环境恢复**：SSH Tunnel + MCP Server + Ollama 全部确认就绪
2. **修复 Ollama 僵死**：旧进程(PID 15713)占端口但无响应，kill 后重启
3. **补全测试标签**：33 张图只有 25 个标签，Python 脚本补生成 8 个随机标签
4. **修复模型权重损坏**：
   - `yolov8n.pt` 仅 380K（应为 6.3MB），从 `/home/kly/yolov8n.pt` 正确副本覆盖
   - `yolo26n.pt` 仅 496K（应为 5.3MB），AMP 检查时 RuntimeError: PytorchStreamReader failed，从 `/home/kly/yolo26n.pt` 覆盖修复
5. **训练成功完成** 🎉：
   - yolov8n 3 epochs，GPU 1 (TITAN X)，2.19GB 显存
   - 最终 mAP50: 0.0265（随机标签，数值正常偏低）
   - 模型保存: `runs/detect/train5/weights/best.pt` (6.2MB)
6. **Agent 交互验证**：
   - scan_dataset ✅（需明确指定 img_dir/label_dir）
   - check_gpu_status ✅（正确识别 GPU 0 忙碌 / GPU 1 空闲）
   - start_training + HITL 确认 ✅（高风险操作拦截正常）
   - check_training_status ✅（能报告运行中/已停止/退出码）

**发现的问题：**

| 问题 | 根因 | 修复方式 |
|------|------|----------|
| 训练 AMP 检查崩溃 | `yolo26n.pt` 损坏（服务器无法联网下载） | 手动复制正确权重到工作目录 |
| 训练模型加载失败 | `yolov8n.pt` 损坏（同上） | 手动复制正确权重 |
| scan 报 78 图 0 标签 | Agent 把整个 `test_dataset/` 传给 `img_dir`，递归扫描了子目录 | 需在 System Prompt 或 Tool docstring 引导 Agent 传 images/ 子目录 |
| Ollama 僵死 | 旧进程卡住占端口但不响应 | kill 重启 |

**教训：**
- 服务器无法联网时，所有模型权重必须预先准备好完整文件
- scan_dataset 的 `img_dir` 参数语义需要更清晰的文档

---

## 八、给下一个会话的指引

### ⚠️ 第一步：先建立 SSH 连接

```powershell
# 必须先开 SSH Tunnel，再做其他任何操作
ssh -L 8080:127.0.0.1:8080 -L 11434:127.0.0.1:11434 yolostudio
```

### 检查 MCP 是否在跑

在已连接的 SSH 终端里：
```bash
ss -tlnp | grep 8080
# 如果没有输出，启动 MCP：
cd /home/kly/yolostudio_agent_proto
nohup /home/kly/miniconda3/envs/yolostudio-agent-server/bin/python -m agent_plan.agent.server.mcp_server > /tmp/mcp.log 2>&1 &
```

### 跑 CLI

```powershell
cd D:\yolodo2.0
D:\yolodo2.0\agent_plan\agent\.venv\Scripts\python.exe agent_plan\agent\client\cli.py
```

### 遗留待办（优先级排序）

1. ~~🔴 **训练完整闭环**~~：✅ 已通过！启动 → 查状态 → 完成
2. 🔴 **scan_dataset 路径引导**：修改 System Prompt 或 Tool docstring，引导 Agent 自动推导 `images/` 和 `labels/` 子目录
3. 🟡 **latest_metrics 解析器**：修正 `train_log_parser.py` 的正则匹配 YOLO 实际输出格式（含 ANSI 转义码）
4. 🟡 **generate_yaml Tool**：暴露 core 的 `generate_yaml` 为 MCP Tool
5. 🟡 **预下载常用模型权重**：编写脚本预下载 yolov8n/yolo26n 等到服务器工作目录，避免训练时联网失败
6. 🟢 **MCP 启动脚本**：systemd 或 supervisor 配置
7. 🟢 **System Prompt 精调**：增加数据集目录结构约定说明
8. 🟢 **Ollama 守护进程化**：避免手动启停

### 关键文件速查

| 要做什么 | 看哪个文件 |
|---------|----------|
| 了解项目全局 | `doc/project_summary.md`（本文档） |
| 了解技术方案 | `doc/yolostudio_agent_feasibility.md` |
| 改 MCP Tool | `agent/server/tools/data_tools.py` 或 `train_tools.py` |
| 改训练逻辑 | `agent/server/services/train_service.py` |
| 改 GPU 检测 | `agent/server/services/gpu_utils.py` |
| 改 Agent 行为 | `agent/client/agent_client.py` |
| 改 CLI 交互 | `agent/client/cli.py` |
| 查真实 API | `D:\yolodo2.0\core\data_handler\_*.py` |
| 查训练 API | `D:\yolodo2.0\core\train_handler.py` |

---

## 九、Git 状态

```
86e8e8c  init: Phase 1 完成 + Phase 2 骨架代码
297f71e  Phase 1+2 验收完成: GPU隔离/MCP启动/Tool验证/SSH免密
8d11b3d  feat: GPU 动态检测 + 设备校验
2aa8425  simplify: 去掉 MIN_TRAIN_FREE_MB 硬门槛
8d55276  fix: split_dataset 补充 mode 合法值说明 + 返回 output_dir 绝对路径
xxxxxxx  fix: _trim_history 保持 tool_call/ToolMessage 配对 + train_service 自动搜索 yolo
```

**服务器代码已同步**（截止 2026-04-08 20:07），MCP Server PID 25915。

**本次修复的服务器文件：**
- `/home/kly/yolostudio_agent_proto/yolov8n.pt` → 从 `/home/kly/yolov8n.pt` 覆盖（6.3MB）
- `/home/kly/yolostudio_agent_proto/yolo26n.pt` → 从 `/home/kly/yolo26n.pt` 覆盖（5.3MB）
- 测试标签补全：`/home/kly/test_dataset/labels/` 补生成 8 个随机标签文件

**训练产出：** `/home/kly/yolostudio_agent_proto/runs/detect/train5/weights/best.pt`

