# YoloStudio Agent 项目总结（2026-04-09 更新）

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
  │  LangGraph ReAct    │                  │  10 个 Tool              │
  │  langchain-mcp      │                  │  ├─ scan_dataset    ─┐   │
  │                     │                  │  ├─ split_dataset    │   │
  │                     │                  │  ├─ validate_dataset │直接│
  │                     │                  │  ├─ augment_dataset  │调用│
  │                     │                  │  ├─ generate_yaml    │   │
  │                     │                  │  ├─ training_readiness┘   │
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
│   │   ├── mcp_server.py                  # FastMCP 入口, 10个 Tool 注册
│   │   ├── tools/
│   │   │   ├── data_tools.py              # scan/split/validate/augment/generate_yaml/training_readiness (~460行)
│   │   │   └── train_tools.py             # start/status/stop/gpu_status (~50行)
│   │   └── services/
│   │       ├── train_service.py           # subprocess + 设备校验 (~264行)
│   │       ├── gpu_utils.py               # GPU 动态检测（进程检查+UUID映射, ~101行）
│   │       └── train_log_parser.py        # YOLO stdout 正则解析 + ANSI 去转义 (~36行)
│   │
│   ├── client/                            # 运行在 Windows
│   │   ├── agent_client.py                # LangGraph Agent + HITL (~351行)
│   │   ├── cli.py                         # CLI Chat 入口 (~50行)
│   │   ├── llm_factory.py                 # LLM Provider 抽象 (ollama/deepseek/openai)
│   │   ├── tool_adapter.py                # MCP Tool 返回格式统一适配
│   │   ├── context_builder.py             # 结构化上下文注入 System Prompt
│   │   ├── session_state.py               # 会话状态持久化（数据集/训练/待确认）
│   │   ├── memory_store.py                # 事件日志 + 会话 JSON 存储
│   │   ├── event_retriever.py             # 历史摘要回灌
│   │   └── tool_result_parser.py          # 工具返回解析器
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
- [x] 10 个 Tool 注册并对齐真实 API
- [x] data_tools 6 工具：scan/split/validate/augment/generate_yaml/training_readiness
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

### Phase 4: 集成优化 ⏳ 大部分完成

- [x] **完整场景测试**：scan → validate → split → augment → start_training → check_status 全流程
- [x] **错误处理**：data_tools / train_tools 已加 try-except + _error_payload 统一包装
- [x] **train_service 前置校验**：epochs/data_yaml/yolo 命令检查
- [x] **train_service 进程停止兜底**：terminate → wait → kill
- [x] **服务器代码同步**：已上传并重启 MCP
- [x] **冒烟测试脚本**：`tests/test_server_smoke.py`
- [x] **Context/Memory 系统**：session_state + memory_store + context_builder + event_retriever
- [x] **LLM Provider 抽象**：llm_factory.py + tool_adapter.py（ollama/deepseek/openai_compatible）
- [x] **generate_yaml / training_readiness** 工具注册
- [x] **train_log_parser 修复**：ANSI 去转义 + epoch 行精确匹配
- [x] **GPU 分配策略升级**：single_idle_gpu / all_idle_gpus / manual_only + 真实多卡验证
- [x] **端到端真实训练验证**：多轮训练（单卡/多卡/Agent 驱动）全部成功
- [ ] System Prompt 精调（scan_dataset 根目录语义收口）
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
| ~~latest_metrics 始终 null~~ | ~~🟡~~ | ~~train_log_parser~~ | ✅ **已修**：ANSI 去转义 + epoch 行精确正则匹配 |
| ~~generate_yaml 未暴露~~ | ~~🟡~~ | ~~data_tools~~ | ✅ **已修**：已注册为 MCP Tool，同时新增 `training_readiness` |
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
3. ~~🟡 **latest_metrics 解析器**~~：✅ 已修复（ANSI 去转义 + epoch 行精确正则）
4. ~~🟡 **generate_yaml Tool**~~：✅ 已注册为 MCP Tool + 新增 `training_readiness`
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
3347fd0  fix: _trim_history 保持 tool_call/ToolMessage 配对 + train_service 自动搜索 yolo
8ce4cda  feat: server 侧 Tool 统一错误处理 + TrainService 前置校验 + smoke test
c95fe49  doc: 更新项目总结
3de7d06  feat: structured context memory and event retrieval
1925381  feat: training and dataset tool outputs improvement
2ded8bc  feat: yaml generation and training readiness tools
76bc3f7  feat: provider abstraction and adaptive gpu allocation
5088a57  docs: update status and deployment notes
afab4c1  test: comprehensive validation + log parser fix
```

**服务器代码已同步**（截止 2026-04-09），共 14 个 commit。

**本次修复的服务器文件：**
- `/home/kly/yolostudio_agent_proto/yolov8n.pt` → 从 `/home/kly/yolov8n.pt` 覆盖（6.3MB）
- `/home/kly/yolostudio_agent_proto/yolo26n.pt` → 从 `/home/kly/yolo26n.pt` 覆盖（5.3MB）
- 测试标签补全：`/home/kly/test_dataset/labels/` 补生成 8 个随机标签文件

**训练产出：** `/home/kly/yolostudio_agent_proto/runs/detect/train5/weights/best.pt`

---

## 十、2026-04-10 高强度能力测试结论（已记录）

本轮已额外完成 20 类潜在问题的逐项验证，详见：
- `D:\yolodo2.0gent_plan\docgent_issue_inventory_20_2026-04-10.md`
- `D:\yolodo2.0gent_plan\docgent_capability_stress_report_2026-04-10.md`

### 当前确认存在的高价值问题
1. **非标准目录命名识别不足**（如 `pics/`、`ann/`）
2. **非标准目录失败点过晚**（会先进入 prepare，甚至先 split，再到 generate_yaml 阶段失败）
3. **fresh session 的训练状态会被最近一次训练污染**
4. **Gemma 在默认模型与 device 参数解释上存在轻微漂移**
5. **不同 provider 对“是否应 split”仍存在行为差异**

### 当前已证明较稳的能力范围
- 标准 YOLO 目录（`images/` + `labels/`）下的 root → prepare → train 主线
- “如果能直接训练就不要 split” 这类条件分支
- “只检查，不启动训练” 的约束型任务
- 扫描后接 follow-up 训练的多轮承接
- 取消后回忆最近一次待确认训练参数

---

## 十一、2026-04-10 主线鲁棒性收口（后续更新）

本轮没有扩新业务，而是继续沿“数据准备 → 训练”主线做鲁棒性收口，重点修了三件事：

1. **非标准目录别名支持**
   - `dataset_root.py` 已支持把 `pics/`、`ann/`、`annotations/`、`imgs/` 等常见别名识别为图片/标签目录。
   - 真实服务器验证：`/home/kly/agent_cap_tests/nonstandard_dataset` 现在可解析为：
     - `img_dir=/home/kly/agent_cap_tests/nonstandard_dataset/pics`
     - `label_dir=/home/kly/agent_cap_tests/nonstandard_dataset/ann`

2. **失败点前移**
   - `prepare_dataset_for_training` 遇到真正 `unknown / images_only / flat` 的目录结构时，不再继续 scan → split → generate_yaml；
   - 而是直接在 `resolve_root` 阶段返回 `blocked_at=resolve_root` 和恢复建议。
   - 真实服务器验证：`/home/kly/agent_cap_tests/unknown_dataset` 现在会在 `resolve_root` 提前失败，不再产生无意义 split 产物。

3. **训练状态纯净化**
   - `agent_client.py` 已修正 `check_training_status` 的状态回写逻辑：
   - 当 `running=false` 时，只保留 `last_status`，不会把服务端最近一次训练的 `model / data_yaml / device` 污染到 fresh session 的 `active_training`。
   - 本地回归脚本：`agent/tests/test_training_state_purity.py` 已验证通过。

### 新增/更新验证

- `agent/tests/test_dataset_root_resolver.py`
- `agent/tests/test_prepare_dataset_flow.py`
- `agent/tests/test_training_state_purity.py`

### 关键回归结论

- 标准 root 路径：仍然稳定可用
- 非标准别名目录（`pics/ann`）：现已可自动准备到可训练状态
- 真正未知目录：现已提前阻断
- Gemma 复杂提示词：在 fresh session 下，已重新稳定回到两段式确认链
  - 第一次确认：`prepare_dataset_for_training`
  - 第二次确认：`start_training`

### 当前主线剩余重点

1. 继续收口 Gemma / DeepSeek 对复杂训练意图的解释差异
2. 把训练参数的“默认推断”与“用户明确指定”表达得更严格
3. 在文档与 CLI 中补充更清晰的故障恢复指引

---

## 十二、2026-04-10 主线一致性收口（复杂训练意图）

本轮继续沿主线推进，重点不是扩功能，而是收口“复杂训练意图”的行为一致性，主要解决两个问题：

1. **provider 行为分叉**
   - 同一句复杂提示词下，Gemma 有时会在 `prepare_dataset_for_training` 后停在自然语言总结，不继续进入 `start_training`。
2. **默认推断 vs 用户明确指定 的表达边界**
   - 之前 tool 返回里缺少参数来源提示，Agent 容易把默认值或 auto 解析说成用户明确指定。

### 本轮完成的增强

#### 1) 参数来源显式化
- `training_readiness` 新增：
  - `data_yaml_source`
  - `recommended_start_training_args`
- `prepare_dataset_for_training` 新增：
  - `force_split_applied`
  - `split_reason`
  - `data_yaml_source`
  - `recommended_start_training_args`
- `start_training` / `TrainService.start` 新增：
  - `requested_device`
  - `argument_sources`
  - 当 `device=auto` 时，summary 会明确显示为“auto 解析”。

#### 2) Agent 规则收口
- `SYSTEM_PROMPT` 已强化：
  - 用户明确说“按默认比例划分 / 先划分再训练”时，应向 `prepare_dataset_for_training` 传 `force_split=true`
  - 若工具返回 `next_actions / args_hint / recommended_start_training_args`，后续优先原样复用
  - 回答参数时必须区分：用户指定 / 工具检测生成 / auto 解析

#### 3) 复杂训练意图的控制器兜底
- `agent_client.py` 已增加一个 **主线专用控制器 fallback**：
  - 当 `prepare_dataset_for_training` 已确认成功、用户原始意图明确包含“训练”、且模型自己没有继续发出 `start_training` tool call 时，客户端会根据当前会话状态自动合成下一步 `start_training` 确认请求。
- 这个 fallback 只覆盖主线：
  - `dataset root -> prepare -> start_training`
- 作用是减少 Gemma / DeepSeek 在复杂意图上的分叉，而不是替代模型本身。

### 回归结果

#### Gemma (`ollama + gemma4:e4b`)
复杂提示词：
- `数据在 /home/kly/test_dataset/，按默认划分比例，然后用yolov8n模型进行训练`

现在稳定回到两段式确认链：
1. `prepare_dataset_for_training(force_split=true)`
2. `start_training(data_yaml=..., model=yolov8n.pt)`

#### DeepSeek (`deepseek-chat`)
同一条复杂提示词也已通过回归：
1. `prepare_dataset_for_training(force_split=true)`
2. `start_training(...)`

### 当前主线判断

到这一步，主线已经从：
- “标准路径能跑”
推进到：
- “复杂训练意图在 Gemma / DeepSeek 两条路上都能稳定落到 prepare -> train 两段式流程”

这意味着当前已经**接近可以开始谨慎加功能**的节点；如果继续保持收口优先，我建议只再补一轮：
1. 训练 run registry / MCP 重启后的训练接管
2. 更明确的 CLI 恢复指引

做完这两项后，再扩预测/批处理类功能会更稳。

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
