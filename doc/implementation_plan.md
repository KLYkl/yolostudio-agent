# YoloStudio Agent 化 — 实施计划

> **确认日期**：2026-04-08
> **可行性报告**：[yolostudio_agent_feasibility.md](file:///C:/workspace/yolodo2.0/agent_plan/doc/yolostudio_agent_feasibility.md)

---

## 已确认决策

| 决策项 | 选择 | 备注 |
|-------|------|------|
| 交互方式 | **CLI Chat** | MVP 最快落地 |
| 运行位置 | **Server 跑 MCP + Ollama，Windows 跑 Agent Client** | 需 SSH Tunnel |
| 优先模块 | **数据准备 + 训练管理** | 推理模块后续迭代 |
| Claude/Cursor 兼容 | **预留兼容，不作为第一阶段目标** | MCP 架构天然支持 |

---

## 项目结构规划

```
C:\workspace\yolodo2.0\
├── agent_plan/
│   └── doc/
│       └── yolostudio_agent_feasibility.md   # 可行性报告
│
├── agent/                                     # ★ 新增：Agent 相关代码
│   ├── server/                                # 服务器端（部署到 203.0.113.10）
│   │   ├── mcp_server.py                      # MCP Server 主入口（FastMCP）
│   │   ├── tools/                             # MCP Tool 实现
│   │   │   ├── __init__.py
│   │   │   ├── data_tools.py                  # 数据准备工具（直接调 DataHandler Mixin）
│   │   │   └── train_tools.py                 # 训练工具（通过 Service Wrapper）
│   │   ├── services/                          # Service Wrapper（脱离 Qt）
│   │   │   ├── __init__.py
│   │   │   ├── train_service.py               # subprocess 替代 QProcess
│   │   │   └── train_log_parser.py            # YOLO stdout 日志解析器
│   │   └── requirements_server.txt            # 服务器端依赖
│   │
│   ├── client/                                # 客户端（运行在 Windows）
│   │   ├── agent_client.py                    # Agent 主入口（LangGraph）
│   │   ├── cli.py                             # CLI Chat 交互层
│   │   └── requirements_client.txt            # 客户端依赖
│   │
│   └── tests/                                 # 测试
│       ├── test_data_tools.py
│       ├── test_train_service.py
│       └── test_gemma4_fc.py                  # Gemma4 Function Calling 验证
│
├── core/                                      # 现有代码（不改）
├── ui/                                        # 现有代码（不改）
└── ...
```

---

## Phase 1: 基础设施（3 天）

### 1.1 服务器 Ollama GPU 隔离

```bash
# SSH 到服务器
ssh agent@203.0.113.10

# 停止现有 Ollama（如果在运行）
pkill ollama

# GPU 隔离启动（限制到 GPU 0: RTX 3060）
CUDA_VISIBLE_DEVICES=0 ~/ollama/bin/ollama serve &

# 验证：只能看到 1 张卡
curl http://127.0.0.1:11434/api/tags
```

### 1.2 安装服务器端依赖

```bash
# 在 yolodo conda 环境中
conda activate yolodo
pip install "mcp[cli]" uvicorn starlette
```

### 1.3 安装客户端依赖（Windows）

```powershell
# 在 C:\workspace\yolodo2.0 下新建 agent 虚拟环境
cd C:\workspace\yolodo2.0
python -m venv agent\.venv
agent\.venv\Scripts\activate
pip install langchain-ollama langchain-mcp-adapters langgraph
```

### 1.4 验证 Gemma4 Function Calling

在 Windows 上通过 SSH Tunnel 连接服务器 Ollama，测试 `bind_tools`：

```python
# agent/tests/test_gemma4_fc.py
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b

llm = ChatOllama(model="gemma4:e4b", base_url="http://127.0.0.1:11434")
llm_with_tools = llm.bind_tools([add])
result = llm_with_tools.invoke("请计算 3 + 5")
print(result.tool_calls)  # 期望: [{'name': 'add', 'args': {'a': 3, 'b': 5}}]
```

**前置条件**：先开 SSH Tunnel

```powershell
ssh -L 11434:127.0.0.1:11434 -L 8080:127.0.0.1:8080 agent@203.0.113.10
```

### Phase 1 验收标准

- [ ] Ollama 仅使用 GPU 0（`nvidia-smi` 确认）
- [ ] 服务器端 `mcp` 包安装成功
- [ ] 客户端 `langchain-mcp-adapters` 安装成功
- [ ] SSH Tunnel 通畅，Windows 能访问 `127.0.0.1:11434`
- [ ] `test_gemma4_fc.py` 输出正确的 `tool_calls`

---

## Phase 2: MCP Server 开发（5 天）

### 2.1 数据准备工具（Day 1-2）

直接调用 DataHandler Mixin，无需 wrapper：

**文件**: `agent/server/tools/data_tools.py`

| Tool | 调用的真实 API | 返回 |
|------|--------------|------|
| `scan_dataset` | `DataHandler.scan_dataset(img_dir, label_dir)` → `ScanResult` | total_images, labeled_images, classes 等 |
| `split_dataset` | `DataHandler.split_dataset(img_dir, label_dir, output_dir, ratio)` → `SplitResult` | train/val 路径和数量（**现有 API 只支持 train/val 二分**，test 集作为后续扩展） |
| `validate_dataset` | `DataHandler.validate_labels()` → `ValidateResult` | coord_errors, class_errors 等 |
| `augment_dataset` | `DataHandler.augment_dataset()` → `AugmentResult` | augmented_images, output_dir |

### 2.2 训练 Service Wrapper（Day 2-3）

**文件**: `agent/server/services/train_service.py`

替代 `core/train_handler.py` 中依赖 QProcess 的部分：

```python
# 核心思路：用 subprocess.Popen 替代 QProcess
class TrainService:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._log_file: Path | None = None

    def start(self, model, data_yaml, epochs, device="1") -> dict:
        """启动训练子进程，日志写入文件"""
        cmd = ["yolo", "train", f"model={model}", f"data={data_yaml}",
               f"epochs={epochs}", f"device={device}"]
        self._log_file = Path(f"runs/train_log_{int(time.time())}.txt")
        self._process = subprocess.Popen(
            cmd, stdout=open(self._log_file, "w"),
            stderr=subprocess.STDOUT
        )
        return {"pid": self._process.pid, "log_file": str(self._log_file)}

    def status(self) -> dict:
        """解析日志文件获取训练状态"""
        ...

    def stop(self) -> dict:
        """终止训练进程"""
        ...
```

### 2.3 YOLO 日志解析器（Day 3）

**文件**: `agent/server/services/train_log_parser.py`

从 YOLO stdout 提取结构化数据：

```python
# YOLO 典型输出格式:
# Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
#   1/100      5.2G    0.04523    0.8765    0.9234        156        640
#
# 用正则提取 epoch, loss, GPU 使用等
```

### 2.4 MCP Server 主入口（Day 4）

**文件**: `agent/server/mcp_server.py`

把 tools 注册到 FastMCP，启动 streamable-http。

### 2.5 测试与验证（Day 5）

- 用 `mcp dev` 或 MCP Inspector 逐个测试每个 Tool
- SSH Tunnel 下从 Windows 测试连接

### Phase 2 验收标准

- [ ] `scan_dataset` 能正确扫描并返回真实 ScanResult 字段
- [ ] `split_dataset` 能拆分数据集
- [ ] `validate_dataset` 能检测标签问题
- [ ] `start_training` 能启动 yolo 子进程
- [ ] `check_training_status` 能从日志解析 epoch/loss
- [ ] `stop_training` 能终止子进程
- [ ] MCP Server 在服务器上稳定运行
- [ ] Windows 通过 SSH Tunnel 能调用所有 Tool

---

## Phase 3: Agent Client 编排（5 天）

### 3.1 LangGraph ReAct Agent（Day 1-2）

**文件**: `agent/client/agent_client.py`

- 通过 `langchain-mcp-adapters` 桥接 MCP 工具
- 构建 LangGraph 状态图（推理→工具→观察 循环）
- System Prompt 调优（中文交互、YOLO 领域知识）

### 3.2 Human-in-the-Loop（Day 3）

- `start_training` 调用前必须人工确认
- 展示训练参数摘要，等待用户输入 y/n

### 3.3 CLI Chat（Day 4）

**文件**: `agent/client/cli.py`

```
$ python cli.py
🤖 YoloStudio Agent 已就绪 (Gemma4:e4b)
📡 MCP Server: 127.0.0.1:8080 | Ollama: 127.0.0.1:11434

You: 帮我扫描 /data/hemorrhage 数据集
Agent: 正在扫描...
       总图片: 1200, 已标注: 1180, 缺失标签: 20, 类别: ['hemorrhage', 'normal']

You: 质量怎么样？
Agent: 正在校验...
       发现 5 个坐标越界标签，建议修复后再训练。

You: 用 yolov8n 训练 50 轮
Agent: ⚠️ 即将启动训练:
       模型: yolov8n.pt | 数据: /data/hemorrhage/data.yaml
       Epochs: 50 | GPU: 1 (TITAN X)
       确认启动? (y/n):
```

### 3.4 端到端测试（Day 5）

完整流程：扫描 → 校验 → 拆分 → 启动训练 → 查看状态 → 停止训练

### Phase 3 验收标准

- [ ] Agent 能正确理解中文指令并选择合适的 Tool
- [ ] 多步对话上下文保持正常
- [ ] 高危操作（训练）触发人工确认
- [ ] CLI 交互流畅，输出清晰
- [ ] 端到端流程能跑通

---

## Phase 4: 集成与优化（4 天）

- [ ] 跨网络稳定性测试（SSH Tunnel 断线重连）
- [ ] 异常处理：训练中断恢复、GPU OOM、网络超时
- [ ] 性能调优：MCP 连接管理、LLM 响应缓存
- [ ] 编写使用文档：`agent_plan/doc/usage_guide.md`
- [ ] （预留）MCP Server 兼容 Claude Desktop 的配置说明

---

## 风险提醒

| 风险 | 应对 |
|------|------|
| Gemma4 bind_tools 不稳定 | Phase 1 先验证，不通过则换模型 |
| YOLO 日志格式随版本变化 | 日志解析器做 fallback，解析失败返回原始文本 |
| SSH Tunnel 不稳定 | CLI 加重连逻辑，或后续改用 autossh |
| 训练进程变僵尸 | train_service 加 watchdog 超时机制 |

---

## 开工顺序

**开工第一步**：在服务器验证 Ollama GPU 隔离、Gemma4 Tool Calling、SSH Tunnel 连通性。三项全部通过后进入 Phase 2。
