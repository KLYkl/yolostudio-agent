# YoloStudio Agent 架构学习手册（面向初学者，扩展版）

> 文件定位：这是 `C:\workspace\yolodo2.0\agent_plan` 里的**学习型文档**，不是单纯项目总结。
> 目标：把这个项目当成一个 Agent 工程样本，用它来解释"Agent 系统为什么这样设计、解决了什么问题、还差什么"。

---

## 0. 你应该怎么使用这份文档

这份文档不是给"只想知道当前能不能用"的人看的。它是给下面这类读者准备的：

- 想学 Agent 开发，但不想停留在概念层
- 想知道一个真实项目为什么要引入 MCP、LangGraph、Memory、HITL
- 想从"项目演进历史"反推技术决策
- 想知道一个工程化 Agent 和"普通聊天机器人"到底差在哪

推荐阅读顺序：

1. 先看 **第 1 章到第 3 章**，建立全局印象
2. 再看 **第 4 章到第 8 章**，理解每层技术是干什么的
3. 再看 **第 9 章到第 12 章**，理解项目是怎么演化出来的
4. 最后看 **第 13 章到第 16 章**，把它变成你的学习地图

---

## 1. 用一句话理解这个项目

这个项目不是"让大模型直接替你训练 YOLO"。

它更准确的定义是：

> **用一个会调用工具的 Agent，把"数据准备 → 训练管理"这条 YOLO 工作流包装成自然语言可操作的系统。**

用户看到的是一句话：

- "扫描这个数据集"
- "先按默认比例划分，然后训练"
- "现在有没有训练在跑"
- "如果有训练在跑就停掉，没有就只告诉我状态"

系统内部做的却是很多层动作：

- 理解意图
- 选择工具
- 调真实代码
- 记录上下文
- 在高风险点暂停
- 让人确认后继续
- 在服务重启后尝试接管训练状态

这就是为什么这个项目适合拿来学 Agent：

> 它不是"会聊天的模型"，而是"由模型驱动、但受系统约束的执行系统"。

![YoloStudio Agent 系统架构全景](images/yolostudio_architecture.png)

---

## 2. 先建立 3 个最重要的认知

### 2.1 认知一：Agent != 聊天机器人

![聊天机器人 vs Agent 系统对比](images/chatbot_vs_agent.png)

> 上图是通用概念图。下面用 YoloStudio 的真实场景解释差异：

很多初学者一说 Agent，脑中想到的是：

- 一个更聪明的聊天框
- 一个能回答更多问题的模型

但在工程里，Agent 更像：

> **一个用自然语言作为入口、用模型做规划、用工具做执行、用状态系统做记忆的软件系统。**

也就是说：

- 聊天机器人重点在"回答"
- Agent 重点在"完成任务"

用图来说明这个差异：

```mermaid
graph LR
    subgraph Chatbot["Chatbot - 聊天机器人"]
        A1["User: How to train YOLO?"] --> B1["Model: You should... 只能回答文字"]
    end

    subgraph AgentSystem["Agent System - YoloStudio Agent"]
        A2["User: 帮我训练 yolov8n"] --> B2["Model: 需要先扫描数据集"]
        B2 --> C2["调用 scan_dataset"]
        C2 --> D2["调用 prepare_dataset"]
        D2 --> E2["调用 start_training"]
        E2 --> F2{"还需要更多步骤?"}
        F2 -->|Yes| B2
        F2 -->|No| G2["汇报: 训练已启动"]
    end
```

这两者的核心差别不是"模型多强"，而是：

- 有没有工具层
- 有没有状态层
- 有没有流程控制
- 有没有风险控制

---

### 2.2 认知二：模型是"大脑"，不是"手脚"

在这个项目里，模型的作用主要是：

- 理解用户说的话
- 判断应该调用哪个工具
- 决定工具调用顺序
- 在工具返回后进行总结

真正干活的是：

- 数据扫描逻辑
- 数据集划分逻辑
- YAML 生成逻辑
- GPU 检测逻辑
- 训练启动 / 查询 / 停止逻辑

也就是说：

> **LLM 负责想，tools/service 负责做。**

```mermaid
graph LR
    subgraph Brain["Brain - LLM Gemma/DeepSeek"]
        P1["1. 理解意图"] --> P2["2. 选择工具"] --> P3["3. 决定顺序"]
    end

    P3 --> T1["scan_dataset"]
    P3 --> T2["split_dataset"]
    P3 --> T3["generate_yaml"]
    P3 --> T4["start_training"]
    P3 --> T5["check_status"]

    T1 --> P4["4. 总结结果"]
    T2 --> P4
    T3 --> P4
    T4 --> P4
    T5 --> P4
```

这是一个非常关键的 Agent 工程原则。

---

### 2.3 认知三：Agent 成败不只看模型，更看系统设计

现实里，很多 Agent 做不稳，不是因为模型不够聪明，而是因为：

- tool 设计太底层
- 数据语义和用户语言不一致
- 上下文没有结构化
- 高风险动作没有 HITL
- 运行状态只放在内存里

所以真正的 Agent 工程，实际上是在解决：

```text
模型规划能力
× 工具抽象质量
× 记忆结构设计
× 运行时流程控制
× 恢复/观测能力
```

这个项目一路演进下来，最值得学习的，恰恰就是这一整套平衡过程。

---

## 3. 当前系统长什么样：整体架构

### 3.1 总体分层图

![YoloStudio Agent 分层架构](images/architecture_layers_cartoon.png)

```mermaid
flowchart LR
    U["用户输入 CLI"]
    CLI["CLI 层 - cli.py"]
    AC["Agent Client - agent_client.py"]
    CTX["上下文层 Context: SessionState + Memory"]
    ORCH["编排层 Orchestration: LangGraph ReAct"]
    LLM["LLM Provider: Ollama / DeepSeek"]
    MCP["MCP Client: langchain-mcp"]
    SERVER["MCP Server: FastMCP :8080"]
    TOOLS["工具层 Tools: data + train"]
    SVCS["服务层 Services: train_service + gpu"]
    CORE["真实能力层: DataHandler + YOLO + GPU"]

    U --> CLI
    CLI --> AC
    AC --> CTX
    AC --> ORCH
    ORCH --> LLM
    ORCH --> MCP
    MCP --> SERVER
    SERVER --> TOOLS
    TOOLS --> SVCS
    SVCS --> CORE
    TOOLS --> CTX
```

### 3.2 为什么要这么分层？

因为如果不分层，最后很容易变成这种混乱结构：

```text
聊天输入 -> 模型 -> if/else -> 直接调代码 -> 直接开训练 -> 结果乱塞回对话
```

这种结构的问题是：

- 很难换模型
- 很难远程部署
- 很难定位问题
- 很难加 HITL
- 很难做状态恢复

所以我们把系统拆成层，每层只负责一件事：

| 层 | 负责什么 |
|---|---|
| CLI 层 | 用户入口 |
| Agent Client 层 | 管理一次对话与一次任务流 |
| 上下文层 | 记住当前状态与历史事件 |
| 编排层 | 管理"模型 → 工具 → 继续/暂停"的流程 |
| LLM Provider 层 | 统一接入不同模型 |
| MCP 层 | 把能力标准化成工具调用 |
| 工具层 | 对外暴露稳定接口 |
| 服务层 | 执行真正业务逻辑 |
| 真实能力层 | 数据处理、训练、文件系统、GPU |

这就是"工程化 Agent"和"脚本拼接"的本质区别。

### 3.3 项目目录结构全览

下面这张图完整展示了代码仓库中每个文件的位置和作用：

```mermaid
graph TD
    ROOT["agent_plan/"] --> CLIENT["agent/client/"]
    ROOT --> SERVER["agent/server/"]
    ROOT --> TESTS["agent/tests/"]
    ROOT --> MEM["memory/"]

    CLIENT --> CLI_F["cli.py - 命令行入口"]
    CLIENT --> AC_F["agent_client.py - 核心协调器"]
    CLIENT --> LLM_F["llm_factory.py - 模型工厂"]
    CLIENT --> SS_F["session_state.py - 会话状态"]
    CLIENT --> MS_F["memory_store.py - 持久化"]
    CLIENT --> CB_F["context_builder.py - 上下文"]
    CLIENT --> ER_F["event_retriever.py - 事件检索"]
    CLIENT --> TA_F["tool_adapter.py - 格式适配"]

    SERVER --> MCP_F["mcp_server.py - MCP 入口"]
    SERVER --> DT_F["data_tools.py - 6个数据工具"]
    SERVER --> TT_F["train_tools.py - 4个训练工具"]
    SERVER --> CT_F["combo_tools.py - 组合工具"]
    SERVER --> DR_F["dataset_root.py - 目录解析"]
    SERVER --> GU_F["gpu_utils.py - GPU 管理"]
    SERVER --> TS_F["train_service.py - 训练服务"]
    SERVER --> TLP_F["train_log_parser.py - 日志解析"]

    TESTS --> TEST_F["26 个测试文件"]

    MEM --> SESS_F["sessions/ - 状态 JSON"]
    MEM --> EVT_F["events/ - 事件 JSONL"]
```

### 3.4 代码量分布

| 模块 | 文件数 | 总行数 | 说明 |
|---|---|---|---|
| Client 层 | 9 | ~1,100 | 对话管理、状态、记忆、上下文 |
| Server Tools | 3 | ~740 | 对外暴露的 MCP 工具 |
| Server Services | 4 | ~910 | 业务逻辑实现 |
| MCP 入口 | 1 | 39 | 工具注册 |
| 测试 | 26 | ~3,500+ | 冒烟/主线/压力/回归 |
| **合计** | **43** | **~6,300** | 不含文档 |

---

## 4. 部署视角：它是怎么跑起来的

### 4.1 部署拓扑图

```mermaid
flowchart LR
    subgraph Win["Windows 本地"]
        CLI["cli.py 命令行"]
        CLIENT["agent_client.py 协调器"]
        MEM["memory/ 记忆存储"]
    end

    subgraph Tunnel["SSH Tunnel 加密隧道"]
        T1[":8080 - MCP 工具服务"]
        T2[":11434 - Ollama 模型服务"]
    end

    subgraph Server["GPU 服务器 - 203.0.113.10"]
        MCPS["FastMCP Server 工具注册"]
        TOOLS["MCP Tools 11个工具"]
        OLLAMA["Ollama 本地模型"]
        YOLO["YOLO 训练进程"]
        GPU["GPU / nvidia-smi 状态检测"]
        FILES["数据集 / 日志 / YAML 配置"]
    end

    CLI --> CLIENT
    CLIENT --> MEM
    CLIENT --> Tunnel
    Tunnel --> MCPS
    Tunnel --> OLLAMA
    MCPS --> TOOLS
    TOOLS --> YOLO
    TOOLS --> GPU
    TOOLS --> FILES
```

### 4.2 这张图说明了什么？

它说明系统有两个关键分离：

#### 分离 1：Agent Client 和真实执行环境分离

- Windows 上跑的是交互层
- 服务器上跑的是执行层

好处：

- 你本地不用背训练环境
- 服务器负责真正 GPU 与数据
- 本地只负责对话和状态组织

#### 分离 2：模型来源和训练资源分离

这个项目现在已经支持：

- 本地/服务器上的 Ollama 模型
- DeepSeek 这类 API provider
- OpenAI-compatible provider

这意味着：

> **训练用什么 GPU，不应该再由"模型部署方式"硬编码决定。**

所以后面才会演化出动态 GPU 策略。

### 4.3 连接方式详解：为什么用 SSH Tunnel

> **知识点：SSH Tunnel 是远程服务最轻量的安全方案**

很多初学者会问："为什么不直接把 MCP Server 开放到公网？"

答案：

1. **安全性**：MCP Server 没有做鉴权，直接公开意味着任何人都能调用你的工具
2. **简单性**：SSH Tunnel 不需要额外配置 TLS、token、OAuth 等
3. **多端口复用**：一条 SSH 连接可以同时转发 MCP (8080) 和 Ollama (11434)

```bash
# 一行命令搞定两个服务的安全访问
ssh -L 8080:127.0.0.1:8080 -L 11434:127.0.0.1:11434 remote-agent
```

```mermaid
sequenceDiagram
    participant Local as Windows :8080 本地端
    participant SSH as SSH Tunnel 加密隧道
    participant Server as Server :8080 服务器端

    Local->>SSH: HTTP 请求
    SSH->>Server: 转发到 localhost
    Server-->>SSH: 返回响应
    SSH-->>Local: 回传结果

    Note over Local,Server: 全程加密，服务器不需要暴露到公网
```

---

## 5. 典型请求：一次用户输入，系统内部怎么走

### 5.1 简单请求：查询当前训练状态

```mermaid
sequenceDiagram
    participant User as 用户
    participant CLI as CLI 命令行
    participant Agent as AgentClient 协调器
    participant Graph as LangGraph 编排
    participant LLM as LLM 大模型
    participant MCP as MCP Tool 工具
    participant Train as train_service 训练服务

    User->>CLI: 有没有训练在跑?
    CLI->>Agent: chat()
    Agent->>Agent: _trim_history() 截断历史
    Agent->>Agent: context_builder 拼装上下文
    Agent->>Graph: prompt + context 送入编排
    Graph->>LLM: 规划下一步 Reason
    LLM->>LLM: 推理 - 需要查训练状态
    LLM->>MCP: Act - check_training_status
    MCP->>Train: status() 查询
    Train->>Train: _sync_runtime_state() 同步进程态
    Train-->>MCP: 训练状态结果
    MCP-->>Graph: 工具返回结果
    Graph->>LLM: Observe - 观察结果
    LLM-->>Agent: 自然语言总结
    Agent->>Agent: _apply_tool_results() 回写状态
    Agent->>Agent: memory.save_state() 持久化
    Agent-->>CLI: 最终回复
    CLI-->>User: 没有训练 / 训练进行中
```

这是最标准的 **ReAct** 模式：

1. 模型先判断需不需要工具（**Reason**）
2. 工具返回真实结果（**Act**）
3. 模型再基于真实结果回答（**Observe**）

---

### 5.2 复杂请求：数据根目录 → 准备 → 训练

```mermaid
sequenceDiagram
    participant User as 用户
    participant Agent as AgentClient 协调器
    participant LLM as LLM 大模型
    participant Combo as prepare_dataset 组合工具
    participant Data as scan/validate/split/yaml
    participant Train as start_training 训练工具

    User->>Agent: 数据在 /home/agent/zyb，划分后训练
    Agent->>LLM: 解析意图
    LLM->>Combo: 调用 prepare_dataset_for_training

    Note over Agent: HIGH RISK 高风险 - 暂停!
    Agent-->>User: 确认 #1 - 执行数据准备?
    User->>Agent: y 确认

    Combo->>Data: resolve-scan-validate-split-yaml-readiness
    Combo-->>Agent: ready + data.yaml + 训练参数提示

    Note over Agent: 准备完成，用户意图包含训练

    Agent->>Agent: _build_followup_training_request()
    Agent-->>User: 确认 #2 - 启动训练?
    User->>Agent: y 确认

    Agent->>Train: start_training
    Train-->>Agent: pid=12345 / device=1 / log_file
    Agent-->>User: 训练已启动!
```

### 5.3 这条链为什么重要？

因为它正好是这个项目最典型、也最容易暴露问题的主线：

- 用户说的是 dataset root
- 工具实际需要的是 img_dir / label_dir / data_yaml
- 模型需要规划多个步骤
- 中间既有"会改数据"的动作，又有"会开长任务"的动作

所以这个项目很多关键设计，都是围绕这条链不断补出来的。

---

## 6. 代码结构总表：每个文件到底在做什么

### 6.1 客户端层

| 文件 | 行数 | 作用 | 为什么要有它 |
|---|---|---|---|
| `cli.py` | 91 | 命令行入口 + 斜杠命令 | 让用户能直接在本地交互 |
| `agent_client.py` | 583 | Agent 主协调器 | 串起 LLM、Graph、Memory、HITL、Tool 结果回写 |
| `llm_factory.py` | 72 | 模型工厂 | 解耦 Ollama / DeepSeek / OpenAI-compatible |
| `session_state.py` | 78 | 结构化会话状态 | 保存当前 dataset、training、pending confirmation |
| `memory_store.py` | 53 | 状态和事件落盘 | 让状态不只存在内存中 |
| `context_builder.py` | 68 | prompt 组装器 | 控制喂给模型的上下文结构 |
| `event_retriever.py` | 102 | 历史摘要与事件检索 | 防止长对话只靠原始历史 |
| `tool_adapter.py` | 49 | Tool 消息格式适配 | 兼容 DeepSeek/OpenAI-compatible 的消息格式要求 |
| `tool_result_parser.py` | 25 | 统一解析 tool result | 把工具返回转成结构化 dict |

---

### 6.2 服务端层

| 文件 | 行数 | 作用 | 为什么要有它 |
|---|---|---|---|
| `mcp_server.py` | 39 | MCP 注册入口 | 把 tools 暴露成 MCP 能调用的接口 |
| `data_tools.py` | 501 | 数据工具接口（6个） | 把数据处理能力包装成 LLM 可调工具 |
| `train_tools.py` | ~100 | 训练工具接口（4个） | 把训练能力包装成 LLM 可调工具 |
| `combo_tools.py` | 140 | 高层组合工具 | 降低复杂任务对模型规划能力的依赖 |
| `dataset_root.py` | 265 | 数据集根目录解析 | 解决"用户说 root，但代码要 img_dir/label_dir" |
| `gpu_utils.py` | 111 | GPU 状态与策略 | 按真实占用决定 auto 设备行为 |
| `train_service.py` | 499 | 训练生命周期管理 | 启动、查询、停止、重启接管训练 |
| `train_log_parser.py` | 36 | 日志解析 | 把 YOLO 训练日志转成机器可理解状态 |

---

### 6.3 完整工具清单：YOLO 数据处理管线

![数据处理流水线](images/data_pipeline_cartoon.png)

当前注册到 MCP 的 11 个工具：

```mermaid
graph LR
    PREPARE["prepare_dataset_for_training"] -.->|"1"| SCAN["scan_dataset"]
    PREPARE -.->|"2"| VALIDATE["validate_dataset"]
    PREPARE -.->|"3"| SPLIT["split_dataset"]
    PREPARE -.->|"4"| YAML["generate_yaml"]
    PREPARE -.->|"5"| READY["training_readiness"]
    AUGMENT["augment_dataset"]
    START["start_training"]
    STATUS["check_training_status"]
    STOP["stop_training"]
    GPU["check_gpu_status"]
```

> **工具分类说明：**
> - **数据工具** (6个)：scan_dataset / validate_dataset / split_dataset / augment_dataset / generate_yaml / training_readiness
> - **训练工具** (4个)：start_training / check_training_status / stop_training / check_gpu_status
> - **组合工具** (1个)：prepare_dataset_for_training（内部依次调用 1→5 五个数据工具）

### 6.4 工具的风险分级

```mermaid
graph LR
    subgraph LowRisk["低风险 LOW RISK - 自动执行"]
        A1["scan_dataset 扫描"]
        A2["validate_dataset 校验"]
        A3["training_readiness 检查"]
        A4["check_training_status 查状态"]
        A5["check_gpu_status GPU"]
        A6["generate_yaml 配置"]
    end

    subgraph HighRisk["高风险 HIGH RISK - 需要人工确认"]
        B1["start_training 启动训练"]
        B2["split_dataset 划分数据"]
        B3["augment_dataset 数据增强"]
        B4["prepare_dataset 一键准备"]
    end
```

高风险工具的共同特征：

- 会修改文件系统（split、augment）
- 会启动长任务（training）
- 会同时做以上两者（prepare）

---

### 6.5 测试层

测试文件很多，但可以按意图归类：

| 类型 | 例子 | 目的 |
|---|---|---|
| 冒烟测试 | `test_server_smoke.py` | 基本链路是否可用 |
| Provider 测试 | `test_llm_factory.py`、`test_tool_adapter.py` | 模型接入是否兼容 |
| 上下文测试 | `test_long_context_smoke.py`、`test_memory_retriever.py` | Memory 是否起作用 |
| 主线流程测试 | `test_prepare_dataset_flow.py`、`test_complex_prompt_flow.py` | root → prepare → train 主线是否稳定 |
| 系统稳定性测试 | `test_train_run_registry.py` | MCP 重启后训练能否接管 |
| 能力边界测试 | `test_agent_capability_range.py` | 复杂提示词下 Agent 能力范围 |
| 大数据脏数据测试 | `test_zyb_large_dataset_e2e.py` | 真实世界数据鲁棒性 |

---

## 7. 核心技术详解：这些技术为什么会被引入

这一章是最适合"系统学知识"的部分。

### 7.1 MCP：为什么需要"工具协议"

![YoloStudio MCP 工具注册全景](images/mcp_tool_registry.png)

#### 7.1.1 它是什么

MCP（Model Context Protocol）可以理解为：

> **把模型能访问的能力，统一包装成标准协议。**

在这个项目里，它的作用很具体：

- `scan_dataset`
- `validate_dataset`
- `prepare_dataset_for_training`
- `start_training`
- `check_training_status`

这些能力，不再只是 Python 函数，而是变成了**模型可调用的工具接口**。

#### 7.1.2 它解决的问题

如果没有 MCP，常见做法是：

- Agent 代码直接 `import` 业务模块
- 模型通过某些本地 function calling 直接驱动内部代码

这会带来问题：

- 强耦合
- 难以远程部署
- 难以跨模型/跨客户端复用
- 未来难以扩展成标准服务

```mermaid
graph LR
    subgraph NoMCP["没有 MCP 时"]
        A1["客户端代码"] -->|"直接 import"| B1["数据处理"]
        A1 -->|"直接 import"| C1["训练管理"]
        A1 -->|"直接 import"| D1["GPU 检测"]
        style A1 fill:#ffcccc
        style B1 fill:#ffcccc
        style C1 fill:#ffcccc
        style D1 fill:#ffcccc
    end

    subgraph WithMCP["有 MCP 时"]
        A2["Agent Client 客户端"] -->|"标准协议"| MCP2["MCP Server 服务器"]
        MCP2 --> B2["数据处理"]
        MCP2 --> C2["训练管理"]
        MCP2 --> D2["GPU 检测"]
        style A2 fill:#ccffcc
        style MCP2 fill:#ccffcc
    end
```

#### 7.1.3 MCP 在代码里长什么样

```python
# mcp_server.py - 只需 3 行就能注册一个工具
mcp = FastMCP("yolostudio", host="127.0.0.1", port=8080)
mcp.tool()(scan_dataset)          # Python 函数 → MCP 工具
mcp.tool()(start_training)        # Python 函数 → MCP 工具
```

```python
# agent_client.py - 客户端连接
client = MultiServerMCPClient({
    "yolostudio": {
        "transport": "streamable-http",
        "url": "http://127.0.0.1:8080/mcp",
    }
})
tools = await client.get_tools()  # 远程获取所有可用工具
```

#### 7.1.4 初学者应该记住什么

> **MCP 的核心价值，不是"能不能调用函数"，而是"能不能把能力变成稳定接口"。**

这是 Agent 系统真正工程化的第一步。

---

### 7.2 FastMCP：为什么不用自己手搓一个协议服务器

FastMCP 的作用是：

- 让你不用自己写底层协议
- 直接把 Python 函数注册成 MCP 工具

这个项目里：

- `mcp_server.py` 是入口
- `mcp.tool()(xxx)` 把函数注册出去

如果手搓，成本会高很多，而且容易把精力浪费在：

- 协议细节
- transport 细节
- 错误处理细节

而不是浪费在真正有价值的业务抽象上。

所以这里的知识点是：

> **工程项目优先复用成熟协议实现，把精力留给业务层。**

---

### 7.3 LangGraph：为什么 Agent 需要"编排框架"

#### 7.3.1 它在这个项目里做什么

这里用它做的是：

- ReAct Agent
- Tool 调用
- `interrupt_before=["tools"]`
- 人工确认恢复
- 统一 graph 状态流

#### 7.3.2 如果没有它会怎样

你也可以自己写一个大循环：

```python
while True:
    llm_reply = call_model(...)
    if wants_tool:
        call_tool()
    if high_risk:
        ask_user_confirm()
```

但很快会遇到：

- 当前在哪个步骤？
- 暂停后怎么恢复？
- 上一轮 graph state 怎么保存？
- 哪些 tool 要自动继续、哪些要停？

```mermaid
stateDiagram-v2
    [*] --> Idle: 空闲
    Idle --> Planning: 用户输入
    Planning --> NeedsConfirm: 高风险工具
    Planning --> Completed: 低风险完成
    NeedsConfirm --> Cancelled: 用户拒绝
    NeedsConfirm --> Approved: 用户确认
    Approved --> Planning: 继续执行
    Planning --> Completed: 全部完成
    Cancelled --> [*]
    Completed --> [*]
```

这就是为什么 Agent 项目里，经常会引入编排层。

#### 7.3.3 LangGraph 在代码里长什么样

```python
# agent_client.py 中创建 ReAct Agent
graph = create_react_agent(
    llm,                            # 模型
    tools,                          # MCP 工具列表
    prompt=SYSTEM_PROMPT,           # 系统规则
    checkpointer=MemorySaver(),     # 状态持久化
    interrupt_before=["tools"],     # 关键: 在工具调用前暂停
)
```

```python
# 暂停后恢复执行
result = await self.graph.ainvoke(
    Command(resume="approved"),     # 用户确认后继续
    config=config
)
```

#### 7.3.4 对初学者最重要的认知

> **Agent 不是"模型 + 几个函数"，而是"状态机 + 模型 + 工具 + 恢复点"。**

LangGraph 本质上是在帮你管理这件事。

---

### 7.4 ReAct：为什么不是直接让模型"一次回答到底"

ReAct 可以粗略理解成：

![ReAct 循环模式 - 推理/执行/观察](images/react_loop_diagram.png)

```mermaid
graph LR
    R["Reason 推理: 需要调 scan_dataset"] --> A["Act 执行: 调用 MCP 工具"]
    A --> O["Observe 观察: 拿到 7027 张图片"]
    O --> R
    O --> F["Final Answer 最终回答"]
```

- 先思考（Reason）
- 再行动（Act）
- 再看结果（Observe）
- 再决定下一步

在这个项目里，用户一句话通常不能直接转成最终答案，尤其是：

- 数据集需要先扫描
- 训练前要先 readiness
- 高风险要先确认

所以这里不能只靠"一次输出最终答案"。

ReAct 的意义是：

> **模型不是一次把所有结论都说完，而是边观察边推进。**

这对真实任务特别关键。

---

### 7.5 HITL：为什么必须有人审确认

![YoloStudio HITL 工作流 - 高风险操作确认](images/hitl_workflow.png)

HITL = Human-in-the-Loop。

在这个项目里，至少这些动作是高风险的：

- `start_training`
- `split_dataset`
- `augment_dataset`
- `prepare_dataset_for_training`

#### 为什么这些危险？

因为它们会：

- 修改数据目录
- 生成新文件
- 启动长任务
- 占用 GPU

如果没有 HITL，模型可能因为：

- 理解偏差
- 参数猜错
- 路径推断错误

而直接造成损失。

#### HITL 在代码里如何实现

```python
# agent_client.py 中的 HITL 实现
HIGH_RISK_TOOLS = {
    "start_training",
    "split_dataset",
    "augment_dataset",
    "prepare_dataset_for_training"
}

async def chat(self, user_text: str, auto_approve: bool = False):
    # ...
    while True:
        pending = self._get_pending_tool_call(config)
        if not pending:
            break
        # 核心判断：是否高风险
        if pending["name"] in HIGH_RISK_TOOLS and not auto_approve:
            self._set_pending_confirmation(thread_id, pending)
            return {
                "status": "needs_confirmation",
                "message": self._build_confirmation_prompt(pending),
            }
        # 低风险 → 自动执行
        result = await self.graph.ainvoke(Command(resume="approved"), config)
```

#### 这里的知识点

> **真实 Agent 要做的不只是"能不能自动执行"，还要决定"哪些动作必须由人兜底"。**

所以这个项目里，HITL 不是装饰，而是主链路的一部分。

---

### 7.6 SessionState：为什么记忆不能只靠聊天历史

#### 7.6.1 它在这个项目里记什么

当前会话状态里主要保存：

- 当前数据集：`dataset_root / img_dir / label_dir / data_yaml`
- 当前训练：`running / model / pid / device / log_file`
- 当前待确认动作：`tool_name / tool_args / thread_id`
- 用户偏好：`default_model / default_epochs / language`

#### 7.6.2 SessionState 数据模型

```mermaid
classDiagram
    class SessionState {
        +session_id: str
        +created_at: str
        +updated_at: str
        +active_dataset: DatasetContext
        +active_training: TrainingContext
        +pending_confirmation: PendingConfirmation
        +preferences: UserPreferences
        +touch()
        +to_dict()
        +from_dict()
    }

    class DatasetContext {
        +dataset_root: str
        +img_dir: str
        +label_dir: str
        +data_yaml: str
        +last_scan: dict
        +last_validate: dict
        +last_split: dict
    }

    class TrainingContext {
        +running: bool
        +model: str
        +data_yaml: str
        +device: str
        +pid: int or None
        +log_file: str
        +started_at: float or None
        +last_status: dict
        +last_start_result: dict
    }

    class PendingConfirmation {
        +thread_id: str
        +tool_name: str
        +tool_args: dict
        +created_at: str
    }

    class UserPreferences {
        +default_model: str
        +default_epochs: int or None
        +language: str
    }

    SessionState --> DatasetContext
    SessionState --> TrainingContext
    SessionState --> PendingConfirmation
    SessionState --> UserPreferences
```

#### 7.6.3 这类信息为什么要结构化？

因为这些信息不是"适合人说给人听"的历史，而是"系统运行必需的状态"。

例如：

- 当前数据集是谁
- 当前训练是否正在跑
- 当前确认的是哪个工具

这些如果只靠聊天记录，很快会变得：

- 模型记不准
- 长上下文成本过高
- 新 session 无法恢复

#### 7.6.4 知识点

> **结构化状态 = 让 Agent 像软件系统；纯聊天历史 = 让 Agent 更像 improvisation。**

工程上必须偏向前者。

---

### 7.7 MemoryStore / EventRetriever：为什么状态之外还要有事件日志

![YoloStudio 双层记忆系统 - 状态快照 + 事件日志](images/memory_state_model.png)

状态能回答：

- 当前是谁
- 现在是什么

但很多问题其实在问：

- 刚才发生了什么
- 为什么会这样
- 最近一次是怎么走到这里的

这时候只靠 state 不够，需要事件流：

```mermaid
graph TD
    subgraph StateLayer["SessionState - 当前快照"]
        S1["dataset: /home/agent/zyb"]
        S2["training: running=true"]
        S3["pending: none"]
    end

    subgraph EventLayer["events.jsonl - 历史记录"]
        E1["T1: scan_dataset - 7027 张图片"]
        E2["T2: 确认请求: prepare"]
        E3["T3: 确认通过: prepare"]
        E4["T4: prepare - 就绪"]
        E5["T5: 确认请求: train"]
        E6["T6: 确认通过: train"]
        E7["T7: start_training - pid=12345"]
    end
```

然后再通过 `EventRetriever` 做：

- 最近事件摘要
- 历史行为压缩
- 针对 session 的记忆回捞

知识点：

> **Agent 的 Memory 至少分成两类：当前状态（state）和过程历史（events）。**

---

### 7.8 ContextBuilder：为什么需要专门"拼 prompt"

很多初学者会把 prompt 理解成一段静态 system prompt。

但这个项目里，真正喂给模型的内容其实是：

```mermaid
graph TD
    SP["固定规则 SYSTEM_PROMPT"]
    SS["当前状态摘要 - ContextBuilder"]
    HD["事件摘要 - EventRetriever"]
    RM["最近消息 - 截断为 12 条"]

    SP --> FINAL["最终消息列表 Messages"]
    SS --> FINAL
    HD --> FINAL
    RM --> FINAL

    FINAL --> LLM["LLM 大模型"]
```

也就是说，prompt 不是固定文案，而是：

> **一个动态拼装的上下文对象。**

这就是 `ContextBuilder` 的价值。

如果没有它，很容易出现：

- prompt 结构失控
- 不同地方拼 prompt 逻辑不一致
- 上下文含义不稳定

---

### 7.9 LLM Provider 抽象：为什么不能把模型写死在 AgentClient 里

#### 一开始的诱惑

最省事的写法通常是：

- 直接在 `agent_client.py` 里 new 一个 `ChatOllama`
- 默认固定模型 `gemma4:e4b`

#### 后果

后面一旦接：

- DeepSeek
- 其他 OpenAI-compatible API
- 本地别的 serving

你会到处改代码。

#### 所以这里怎么做的

通过 `llm_factory.py` 抽象成：

```mermaid
graph TD
    FACTORY["llm_factory.py - build_llm() 模型工厂"]

    FACTORY -->|"本地模型 provider=ollama"| OLLAMA["ChatOllama"]
    FACTORY -->|"云端模型 provider=deepseek"| DS["ChatOpenAI - DeepSeek"]
    FACTORY -->|"兼容接口 provider=openai_compatible"| OAI["ChatOpenAI - 其他 API"]

    OLLAMA --> LLM_IF["统一接口 BaseChatModel"]
    DS --> LLM_IF
    OAI --> LLM_IF

    LLM_IF --> AGENT["AgentClient - 只依赖接口，不依赖具体实现"]
```

配置方式：

```python
@dataclass
class LlmProviderSettings:
    provider: str   # "ollama" / "deepseek" / "openai_compatible"
    model: str      # "gemma4:e4b" / "deepseek-chat" / 任意
    base_url: str
    api_key: str
    temperature: float
```

#### 知识点

> **模型是系统里的可替换部件，不应该变成系统结构本身。**

---

### 7.10 tool_adapter：为什么会有一个"看起来有点奇怪"的适配层

这是一个特别有代表性的真实工程问题。

我们接入 DeepSeek / OpenAI-compatible provider 时，发现：

- 有些模型对 tool message 的 `content` 更严格
- 原始 MCP tool 返回 block/list 结构时会报格式错

```mermaid
graph LR
    subgraph Problem["问题场景"]
        MCP_RET["MCP 原始返回 - block 列表"]
        DS["DeepSeek 期望纯字符串"]
        MCP_RET -->|"格式不匹配"| ERROR["报错!"]
    end

    subgraph Solution["解决方案 - 加适配层"]
        MCP_RET2["MCP 原始返回"]
        ADAPTER["tool_adapter.py 格式转换"]
        DS2["DeepSeek 收到 JSON 字符串"]
        MCP_RET2 --> ADAPTER --> DS2
        DS2 --> OK["成功!"]
    end
```

解决方法不是去改整个系统，而是加一个适配层：

- 把 MCP 的 tool 输出适配成模型想要的字符串格式

这件事的知识点非常经典：

> **不同模型 provider 之间，问题往往不在"能不能调用"，而在"消息协议细节是否一致"。**

所以工程上经常需要 adapter 层。

---

### 7.11 dataset_root resolver：为什么一个"目录解析器"会这么重要

这是这个项目主线里最具教学意义的一个点。

#### 用户说的话 vs 工具需要的

![dataset_root resolver 语义翻译过程](images/dataset_root_resolver.png)

```mermaid
graph LR
    subgraph UserSays["用户说的路径"]
        U1["/data/test_dataset/"]
        U2["C:/datasets/zyb"]
    end

    subgraph ToolNeeds["工具需要的参数"]
        T1["img_dir=.../images"]
        T2["label_dir=.../labels"]
        T3["data.yaml 路径"]
    end

    U1 -->|"语义鸿沟 semantic gap"| GAP["dataset_root.py 解析"]
    GAP --> T1
    GAP --> T2
    GAP --> T3
```

#### resolver 如何工作

```mermaid
flowchart TD
    INPUT["输入路径"]

    INPUT --> CHECK1{"路径存在?"}
    CHECK1 -->|No| ERR1["路径不存在"]
    CHECK1 -->|Yes| CHECK2{"是图片目录?"}

    CHECK2 -->|Yes| TYPE1["images_dir 类型"]
    CHECK2 -->|No| CHECK3{"有 images/ 子目录?"}

    CHECK3 -->|Yes| CHECK4{"有 labels/ 子目录?"}
    CHECK3 -->|No| CHECK5{"有其他图片子目录?"}

    CHECK4 -->|Yes| TYPE2["yolo_standard 标准结构"]
    CHECK4 -->|No| TYPE3["images_only 仅图片"]

    CHECK5 -->|Yes| TYPE4["heuristic 启发式推断"]
    CHECK5 -->|No| CHECK6{"根目录有图片?"}

    CHECK6 -->|Yes| TYPE5["flat 平铺结构"]
    CHECK6 -->|No| TYPE6["unknown 未知结构"]

    TYPE2 --> SPLIT_CHECK{"有 train/val 划分?"}
    SPLIT_CHECK -->|Yes| TYPE7["yolo_split 已划分"]
    SPLIT_CHECK -->|No| TYPE2
```

支持的目录别名：

| 类型 | 支持的名称 |
|---|---|
| 图片目录 | `images`, `imgs`, `jpegimages`, `pics`, `pictures`, `imageset` |
| 标签目录 | `labels`, `annotations`, `label`, `ann`, `anns`, `txt_labels` |

#### 知识点

> **Agent 的难点，经常不在算法，而在把用户自然语言映射成系统真正需要的参数。**

---

### 7.12 combo tool：为什么高层组合工具能极大降低失败率

这是这个项目后期一个很关键的思想升级。

一开始工具都是底层动作：

```mermaid
graph LR
    subgraph Before["改造前: LLM 需要规划 6 步"]
        M1["LLM 模型"] --> S1["scan"] --> S2["validate"] --> S3["split"]
        S3 --> S4["gen yaml"] --> S5["readiness"]
    end
```

这样做的问题是：

- 模型要自己规划很多步
- 任何一步理解错都会失败
- 尤其是较弱模型更容易空白或卡住

于是后来加了：

```mermaid
graph LR
    subgraph After["改造后: LLM 只需 1 步"]
        M2["LLM 模型"] --> COMBO["prepare_dataset_for_training"]
        COMBO -.->|"内部调用"| S1["scan"]
        COMBO -.->|"内部调用"| S2["validate"]
        COMBO -.->|"内部调用"| S3["split"]
        COMBO -.->|"内部调用"| S4["gen yaml"]
        COMBO -.->|"内部调用"| S5["readiness"]
    end
```

这不是为了"偷懒"，而是为了：

> **把复杂度从模型转移到系统。**

这几乎是 Agent 工程里最重要的经验之一。

---

### 7.13 gpu_utils：为什么 GPU 策略不能拍脑袋写死

这块我们踩过很真实的坑。

一开始很容易想：

- 0 卡给 LLM
- 1 卡给训练

但后来发现这不可靠，因为：

- 有时换成 API provider，本地 GPU 不跑模型
- 有时未来可能用 vLLM，多卡 serving
- 有时两张卡都空闲
- 有时某张卡正在被别的进程占用

所以后来改成：

```mermaid
flowchart TD
    INPUT["device=auto 自动选择"]
    INPUT --> QUERY["查询 nvidia-smi 实时状态"]
    QUERY --> POLICY{"当前策略?"}

    POLICY -->|"single_idle_gpu 单卡"| BEST["选空闲内存最大的 GPU"]
    POLICY -->|"all_idle_gpus 多卡"| ALL["选所有空闲 GPU"]
    POLICY -->|"manual_only 手动"| REJECT["拒绝 auto，要求手动指定"]

    BEST --> RESULT["device=1"]
    ALL --> RESULT2["device=0,1"]
    REJECT --> ERROR["返回错误"]
```

三种 GPU 分配策略：

| 策略 | 说明 | 适用场景 |
|---|---|---|
| `single_idle_gpu` | 只选 1 张空闲 GPU | 默认，最安全 |
| `all_idle_gpus` | 选所有空闲 GPU | 多卡训练 |
| `manual_only` | 禁止 auto | 严格控制环境 |

这件事对应的知识点是：

> **资源策略应该依赖运行时真实状态，而不是依赖对部署方式的静态想象。**

---

### 7.14 train_service：为什么要有一层"训练服务"而不是直接在 tool 里开进程

![训练任务生命周期 - 从启动到重启接管](images/training_lifecycle.png)

如果直接在 `train_tools.py` 里把所有事情都做掉，会出现：

- 启动逻辑和工具接口混在一起
- 训练状态、日志、pid 管理混乱
- 后续扩展 stop / reattach / registry 会很痛

所以后来引入 `train_service.py`，让它专门负责：

```mermaid
classDiagram
    class TrainService {
        -_process: Popen
        -_active_pid: int
        -_log_file: Path
        -_start_time: float
        -_resolved_device: str
        -_command: list
        -_reattached: bool
        -_active_registry_path: Path
        -_last_registry_path: Path

        +start() dict
        +status() dict
        +stop() dict
        -_sync_runtime_state()
        -_is_running() bool
        -_resolve_device() tuple
        -_find_yolo_executable() str
        -_write_active_registry()
        -_finalize_active_registry()
        -_load_registry_into_runtime()
        -_validate_inputs()
        -_pid_exists() bool
        -_terminate_pid() tuple
    }
```

知识点：

> **业务长任务通常需要一个专门的 service 层，不应该直接塞进 tool handler。**

---

### 7.15 run registry：为什么"进程内句柄"不够

这点已经在主线上被证明是关键系统项。

#### 一开始的问题

- 训练启动后，`TrainService` 只在内存里记 `_process`
- 只要 MCP 重启，句柄就没了

#### 解决方案：持久化注册表

```mermaid
sequenceDiagram
    participant User as 用户
    participant MCP1 as MCP Server v1 旧实例
    participant Registry as 注册表文件
    participant YOLO as YOLO 训练进程
    participant MCP2 as MCP Server v2 新实例

    User->>MCP1: start_training 启动训练
    MCP1->>YOLO: spawn subprocess pid=12345
    MCP1->>Registry: 写入 active_train_job.json

    Note over MCP1: MCP 重启了!
    MCP1->>MCP1: 进程句柄丢失

    Note over YOLO: 训练仍在服务器上运行...

    User->>MCP2: check_training_status 查状态
    MCP2->>Registry: 读取 active_train_job.json
    MCP2->>MCP2: 发现 pid=12345
    MCP2->>YOLO: os.kill(12345, 0) 检查存活
    YOLO-->>MCP2: 进程存活!
    MCP2->>MCP2: 从注册表重新接管 reattach
    MCP2-->>User: 训练活跃, reattached=true

    User->>MCP2: stop_training 停止训练
    MCP2->>YOLO: SIGTERM 终止信号
    MCP2->>Registry: 写入 last, 删除 active
    MCP2-->>User: 训练已停止
```

`active_train_job.json` 结构：

```json
{
  "pid": 12345,
  "log_file": "/path/to/train_log_1234567890.txt",
  "started_at": 1712345678.9,
  "device": "1",
  "command": ["yolo", "train", "model=yolov8n.pt", "..."],
  "resolved_args": {"model": "yolov8n.pt", "epochs": 100},
  "running": true
}
```

知识点：

> **任务生命周期只要跨越进程边界，就必须有 durable state。**

---

### 7.16 train_log_parser：为什么还要解析日志

训练不是一个瞬时动作，而是长任务。

如果不解析日志，你只能知道：

- 进程在不在
- 日志文件路径是什么

但用户更想知道的是：

- 训练是否还在跑
- 最近有没有指标
- 当前进展大概如何

```python
# YOLO 训练日志原始输出示例
# \x1b[1m  5/100\x1b[0m     3.12G   1.234   0.567   0.890   ...

# train_log_parser.py 提取出：
{
    "epoch": 5,
    "total_epochs": 100,
    "gpu_mem": "3.12G",
    "box_loss": "1.234",
    "cls_loss": "0.567",
    "dfl_loss": "0.890"
}
```

这件事的知识点是：

> **长任务系统里，"日志"不只是给人看的文本，也应该成为机器状态输入。**

---

## 8. 这些模块是如何配合的：结构图细化

### 8.1 Client 层内部关系图

```mermaid
flowchart TD
    CLI["cli.py - 用户入口"] --> AC["agent_client.py - 核心协调"]
    AC --> LLMF["llm_factory.py - 创建模型"]
    AC --> STORE["memory_store.py - 读写状态"]
    AC --> STATE["session_state.py - 数据模型"]
    AC --> BUILDER["context_builder.py - 拼装上下文"]
    AC --> RETRIEVER["event_retriever.py - 历史摘要"]
    AC --> PARSER["tool_result_parser.py - 解析返回"]
    AC --> ADAPTER["tool_adapter.py - 格式适配"]

    BUILDER --> STATE
    BUILDER --> RETRIEVER
    RETRIEVER --> STORE
    STORE --> STATE
```

### 8.2 这里体现了什么设计思想？

- `cli.py` 只负责输入输出，不负责复杂逻辑
- `agent_client.py` 负责串联，而不是把所有逻辑写死在自己内部
- Memory、Context、LLM 接入都被拆分成可独立理解的组件

这是典型的"控制器 + 组件"架构。

---

### 8.3 服务端内部关系图

```mermaid
flowchart TD
    MCP["mcp_server.py - MCP 入口"] --> DT["data_tools.py - 6个数据工具"]
    MCP --> TT["train_tools.py - 4个训练工具"]
    MCP --> CT["combo_tools.py - 组合工具"]

    DT --> ROOT["dataset_root.py - 目录解析"]
    DT --> GPU["gpu_utils.py - GPU 检测"]
    TT --> TRAIN["train_service.py - 生命周期管理"]
    CT --> DT
    TRAIN --> GPU
    TRAIN --> LOG["train_log_parser.py - 日志解析"]
    TRAIN --> YOLO["YOLO CLI / subprocess 训练进程"]
    TRAIN --> REG["active_train_job.json 注册表"]
```

这个结构说明：

- combo tool 不直接自己实现所有逻辑
- train tool 不直接自己管理训练生命周期
- dataset root / gpu / train 是可复用服务能力

这是服务层抽象的价值。

### 8.4 完整的数据流：从用户输入到训练产出

```mermaid
flowchart TD
    USER["用户输入"]

    subgraph CLIENT_SIDE["Windows 本地"]
        CLI["cli.py 命令行"] --> AC["agent_client.py 协调"]
        AC --> CB["上下文拼装 context_builder"]
        AC --> LG["LangGraph ReAct 编排"]
        LG --> LLM["LLM 推理"]
        LLM --> TC["工具调用请求 Tool Call"]
        AC --> HITL{"高风险?"}
        HITL -->|Yes| CONFIRM["暂停等确认"]
        HITL -->|No| EXEC["自动执行"]
        AC --> SS["状态更新 session_state"]
        AC --> MEM["持久化 memory_store"]
    end

    subgraph NETWORK["SSH Tunnel 加密隧道"]
        MCP_CALL["HTTP 请求"]
    end

    subgraph SERVER_SIDE["GPU 服务器"]
        MCPS["FastMCP Server"]
        RESOLVE["目录解析 dataset_root"]
        SCAN["扫描 scan_dataset"]
        VALID["校验 validate"]
        SPLIT["划分 split_dataset"]
        GENYAML["配置 generate_yaml"]
        READY["就绪检查 readiness"]
        GPU_CHECK["GPU 检测 gpu_utils"]
        TRAIN["训练启动 train_service"]
        YOLOP["YOLO 训练进程"]
        LOG_F["训练日志"]
        REG_F["注册表 registry"]
    end

    USER --> CLI
    TC --> MCP_CALL --> MCPS
    MCPS --> RESOLVE --> SCAN --> VALID --> SPLIT --> GENYAML --> READY
    READY --> GPU_CHECK
    GPU_CHECK --> TRAIN --> YOLOP
    TRAIN --> LOG_F
    TRAIN --> REG_F
```

---

## 9. 数据与状态：Agent 到底记了什么

### 9.1 数据存储全景

```mermaid
graph TD
    MemDir["memory/ 目录 - Windows 本地"] --> SJ["sessions/default.json - 当前快照"]
    MemDir --> EJ["events/default.jsonl - 行为时间线"]

    RunsDir["runs/ 目录 - 服务器端"] --> ATJ["active_train_job.json 活跃任务"]
    RunsDir --> LTJ["last_train_job.json 最近任务"]
    RunsDir --> LOGS["train_log_*.txt 训练日志"]
```

### 9.2 events.jsonl 是干什么的

状态只存"当前值"，但 Agent 还需要记"过程"。

当前 events 里会记：

| 事件类型 | 含义 | 触发时机 |
|---|---|---|
| `tool_result` | 工具调用结果 | 每次工具返回 |
| `confirmation_requested` | 请求人工确认 | 高风险工具调用 |
| `confirmation_approved` | 用户确认 | 用户输入 y |
| `confirmation_cancelled` | 用户取消 | 用户输入 n |

你可以把它理解成：

- `SessionState` = 当前快照
- `events.jsonl` = 行为日志

---

## 10. 主线状态机：一次高风险任务如何被控制

### 10.1 HITL 状态机

```mermaid
stateDiagram-v2
    [*] --> Idle: 空闲
    Idle --> Planning: 用户输入
    Planning --> NeedsConfirmation: 高风险工具
    Planning --> Completed: 低风险完成
    NeedsConfirmation --> Cancelled: 用户拒绝
    NeedsConfirmation --> Approved: 用户确认
    Approved --> Planning: 恢复执行
    Planning --> Completed: 全部完成
    Cancelled --> [*]
    Completed --> [*]
```

### 10.2 这张图说明什么？

高风险流程不是：

- 模型想调用就直接执行

而是：

1. 模型提出动作
2. 系统识别是否高风险
3. 如果高风险，就先暂停
4. 用户确认后再继续

这正是"Agent 工程里的控制权边界"。

---

### 10.3 训练任务生命周期图

```mermaid
stateDiagram-v2
    [*] --> NotRunning: 未运行
    NotRunning --> Starting: start_training 启动
    Starting --> Running: 子进程启动成功
    Starting --> FailedToStart: 验证失败
    Running --> Running: check_status 查询
    Running --> Stopping: stop_training 停止
    Running --> Reattached: MCP 重启 + 注册表接管
    Reattached --> Running: 恢复监控
    Stopping --> NotRunning: 已停止
    Running --> Finished: 训练完成
    Finished --> NotRunning
    FailedToStart --> NotRunning
```

这张图对应的知识点是：

> **训练不是一个函数调用，而是一个生命周期对象。**

所以它需要：

- 启动态
- 运行态
- 停止态
- 重启接管态

---

## 11. 项目是怎么演化出来的：按提交看技术决策

![项目演化时间线](images/evolution_timeline_cartoon.png)

下面把全部 22 个提交串成一条完整的技术演化路径。

### 11.1 完整 Commit 时间线

```mermaid
graph TD
    C1["86e8e8c - Phase 1 骨架"]
    C2["297f71e - GPU隔离 + MCP + SSH"]
    C3["8d11b3d - GPU 动态检测"]
    C4["2aa8425 - 移除硬编码阈值"]
    C5["8d55276 - split 模式修复"]
    C6["3347fd0 - trim_history 配对"]
    C7["8ce4cda - 错误处理 + 冒烟测试"]
    C8["c95fe49 - 文档更新"]
    C9["3de7d06 - 结构化记忆"]
    C10["1925381 - 工具输出优化"]
    C11["2ded8bc - yaml生成 + readiness"]
    C12["76bc3f7 - Provider 抽象"]
    C13["5088a57 - 部署文档"]
    C14["afab4c1 - 校验 + 日志解析"]
    C15["7acae6c - dataset_root 解析器"]
    C16["6ed5214 - 状态纯化"]
    C17["f157bfb - GPU + 训练契约"]
    C18["1aee7be - 训练意图一致性"]
    C19["a2e7a65 - run registry 注册表"]
    C20["d6fecd8 - 大数据集压力测试"]
    C21["d25692e - 学习手册 v1"]
    C22["073b48c - 学习手册扩展"]

    C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8
    C8 --> C9 --> C10 --> C11 --> C12 --> C13 --> C14
    C14 --> C15 --> C16 --> C17 --> C18 --> C19 --> C20
    C20 --> C21 --> C22
```

### 11.2 六个演化阶段总结

| 阶段 | 代表提交 | 主要变化 | 学到的知识 |
|---|---|---|---|
| 骨架阶段 | `86e8e8c` → `297f71e` | Phase 1 骨架、最小 server/client、GPU隔离、SSH免密 | **先打通最小链路，再谈完整性** |
| 工具健壮化 | `8d11b3d` → `8ce4cda` | GPU动态检测、错误处理、前置校验、split修复、yolo搜索 | **Tool contract 比"能调用"更重要** |
| 记忆系统 | `3de7d06` → `2ded8bc` | structured context、memory+event、工具输出升级、YAML生成 | **Agent 记忆必须结构化** |
| 抽象层 | `76bc3f7` → `7acae6c` | provider抽象、GPU策略升级、dataset root resolver、preparation flow | **把复杂度从模型转移到系统** |
| 鲁棒性 | `6ed5214` → `1aee7be` | 状态纯净化、训练规则契约、双provider一致性 | **真正困难在"稳定性和一致性"** |
| 准生产 | `a2e7a65` → `d6fecd8` | run registry、MCP重启接管、大数据脏数据压力测试 | **durable state 是准生产分水岭** |

### 11.3 这条演化路径最值得学习的地方

它说明了一个真实 Agent 项目不是"先把所有功能都想完再做"，而是：

1. 先打通最小主线
2. 再补工具契约
3. 再补状态系统
4. 再补抽象层
5. 再补鲁棒性
6. 再补接近投入使用的系统能力

这和很多纯理论教程非常不一样。

---

## 12. 遇到的困难：问题、根因、解决方式

下面这些是真实踩过的坑，不是理论问题。

### 12.1 复杂提示词在 Gemma 下空白输出

#### 表现

用户说：

> "数据在某个目录里，按默认比例划分，然后训练"

Gemma 有时直接空白，没有后续动作。

#### 根因分析

```mermaid
graph TD
    CAUSE1["底层工具太多"] --> EFFECT["模型需规划5步以上"]
    CAUSE2["dataset root 语义模糊"] --> EFFECT
    CAUSE3["没有高层组合工具"] --> EFFECT
    EFFECT --> RESULT["Gemma 规划失败 / 空白输出"]
```

#### 解决方式

- 加 `dataset_root resolver` → 自动翻译用户路径
- 加 `prepare_dataset_for_training` → 把 5 步压缩为 1 步
- 让复杂任务先收敛成两段式：prepare → start_training

#### 学到的知识

> **当复杂任务失败时，先问"是不是系统给模型的任务空间太复杂了"，而不是先怪模型不聪明。**

---

### 12.2 DeepSeek 工具链报消息格式问题

#### 表现

切到 DeepSeek / OpenAI-compatible provider 后，tool 调用路径报消息格式错误。

#### 根因

```python
# MCP 原始返回格式
[{"type": "text", "text": '{"ok": true, ...}'}]

# DeepSeek 期望的 ToolMessage.content 格式
'{"ok": true, ...}'  # 纯字符串
```

#### 解决方式

引入 `tool_adapter.py`，在 provider 边界做格式适配。

#### 学到的知识

> **provider abstraction 不只是"构造不同模型对象"，还包括"处理 provider 协议细节差异"。**

---

### 12.3 长对话后模型越来越乱

#### 表现

会话长了以后：参数漂移、状态记错、历史太重

#### 根因

```mermaid
graph LR
    MSGS["_messages 列表不断增长"] --> P1["上下文越来越长"]
    P1 --> P2["关键信息被淹没"]
    P2 --> P3["模型参数漂移"]
```

#### 解决方式

引入四层记忆架构：

```mermaid
graph TD
    SS["SessionState 结构化状态"] --> CB["ContextBuilder 拼装上下文"]
    MS["MemoryStore 持久化"] --> ER["EventRetriever 事件摘要"]
    ER --> CB
    CB --> TRIM["_trim_history 保留 12 条"]
    TRIM --> LLM["精简上下文送入 LLM"]
```

#### 学到的知识

> **LLM 的长上下文能力不是无限免费的，必须用状态设计去减压。**

---

### 12.4 fresh session 会被旧训练信息污染

#### 表现

新 session 只是查状态，结果把最近一次训练的 model/data_yaml/device 带进来了。

#### 根因

状态回写逻辑过宽——"当前训练"和"最近一次训练"语义混在一起。

#### 解决方式

```python
# 修复前：不管 running 与否，都更新所有字段
elif tool_name == "check_training_status":
    tr.device = result.get("device", tr.device)  # 污染
    tr.pid = result.get("pid", tr.pid)            # 污染

# 修复后：只在 running=true 时更新
elif tool_name == "check_training_status":
    tr.last_status = result
    is_running = bool(result.get("running"))
    tr.running = is_running
    if is_running:  # 只在真正运行时更新
        tr.device = result.get("device", tr.device)
        tr.pid = result.get("pid", tr.pid)
    else:
        tr.pid = None  # 清理
```

#### 学到的知识

> **状态系统里最怕的不是"记不住"，而是"把不该记成当前态的东西记成当前态"。**

---

### 12.5 MCP 重启后训练失联

#### 表现

训练还在跑，但 MCP 重启后 stop/status 无法继续接管。

#### 根因

训练状态只保存在内存 `_process` 句柄里。

#### 解决方式

引入 run registry（详见 7.15 节）。

#### 学到的知识

> **长任务系统必须把运行态持久化，否则一切重启都是"失忆"。**

---

### 12.6 数据集根目录被误当成图片目录

#### 表现

用户说的是 `/data/test_dataset/`，系统却把它当成 `img_dir`，导致递归扫描子目录，图片计数从 33 暴增到 78。

#### 解决方式

引入 `dataset_root.py`（详见 7.11 节）。

#### 学到的知识

> **用户输入语义和程序语义之间，往往需要一个专门的翻译层。**

---

### 12.7 GPU 规则一开始矫枉过正

#### 表现

曾经默认把某张卡固定留给 LLM，把训练多卡直接拒掉。

#### 根因

把某种部署方式误当成永久规则。

#### 解决方式

改成按 `nvidia-smi` 实际占用 + 三种策略（single/all/manual）决策。

#### 学到的知识

> **不要把某次部署的偶然条件写成系统永恒规则。**

---

### 12.8 Gemma 的解释层比执行层更容易失真

#### 表现

Gemma 常常 tool 调得对，但总结时会说过头，或把默认推断说成用户明确指定。

#### 解决方式

```python
# 工具返回现在包含参数来源标注
{
    "argument_sources": {
        "model": "request_or_agent_input",
        "data_yaml": "request_or_tool_output",
        "epochs": "request_or_default",
        "device": "auto_resolved"  # 明确标注是 auto 解析
    }
}
```

#### 学到的知识

> **Agent 的"解释层"要单独治理，不能以为工具调对了，最终表述就一定可靠。**

---

### 12.9 yolo 命令找不到

#### 表现

服务器上有 `yolo`，但 `train_service` 找不到。

#### 根因

`yolo` 安装在 conda 环境中，不在系统 PATH 里。

#### 解决方式

```python
def _find_yolo_executable() -> str | None:
    # 1. 先在 PATH 里找
    yolo_in_path = shutil.which('yolo')
    if yolo_in_path:
        return yolo_in_path

    # 2. 通过 conda env list 找所有环境
    result = subprocess.run(['conda', 'env', 'list'], ...)
    for env_path in env_paths:
        yolo = _resolve_yolo_in_env(env_path)
        if yolo:
            return yolo

    # 3. 搜索常见 conda 安装路径
    for root in [~/anaconda3/envs, ~/miniconda3/envs, ...]:
        ...
```

#### 学到的知识

> **Agent 系统不能假设工具在默认 PATH 中，需要主动搜索运行时环境。**

---

### 12.10 CLI 历史截断导致 INVALID_CHAT_HISTORY 崩溃

#### 表现

长对话后 CLI 直接崩溃，报 `INVALID_CHAT_HISTORY`。

#### 根因

`_trim_history()` 截断时，可能把 `AIMessage(tool_calls=[...])` 保留了，但对应的 `ToolMessage` 被截掉了，导致消息配对不完整。

#### 解决方式

截断时需要保证 tool_call 和 ToolMessage 的完整配对。

#### 学到的知识

> **LLM 框架对消息格式有严格的结构要求，截断历史时不能只看数量，还要看消息配对完整性。**

---

## 13. 当前能力边界：已经能做什么，还不能做什么

### 13.1 当前比较稳的能力

- 标准 YOLO root 的识别（含非标别名如 pics/ann）
- scan / validate / split / augment / generate_yaml
- readiness 判断
- start / status / stop training
- 高风险动作确认（HITL 两段式）
- Ollama 与 DeepSeek 双 provider 主线
- MCP 重启后训练接管
- 大数据集（7000+ 图）主线冒烟

### 13.2 当前明确还不够成熟的点

- 非标准目录虽然变强了，但还不是"什么目录都能懂"
- 大量缺失标签图片还没有稳定提升为强 blocker
- 类名语义保留仍有改进空间
- durable checkpoint 还没升到真正生产级
- tracing / observability / eval 还不够系统化

### 13.3 能力成熟度

| 成熟度 | 能力项 | 说明 |
|---|---|---|
| ✅ **稳定** | 标准数据准备主线 | scan/validate/split/yaml/readiness 全链 |
| ✅ **稳定** | 双 Provider 训练 | Ollama + DeepSeek 主线均已验证 |
| ✅ **稳定** | HITL 人工确认 | 两段式确认机制稳定运行 |
| ✅ **稳定** | 训练生命周期 | start/status/stop 完整链路 |
| ✅ **稳定** | MCP 重启接管 | run registry 持久化注册表 |
| ⚠️ **有限** | 非标目录识别 | 已支持 6 种别名，但不是“什么都能懂” |
| ⚠️ **有限** | 脏数据风险报告 | 能检测但还不是强 blocker |
| ⚠️ **有限** | 类名语义保留 | 已在改进中 |
| ❌ **尚未** | 多用户共享 | 当前单用户 |
| ❌ **尚未** | 鉴权与审计 | 依赖 SSH Tunnel |
| ❌ **尚未** | 平台级评测 | 手动测试为主 |
| ❌ **尚未** | 预测与导出 | 未实现 |

---

## 14. 当前系统与官方/主流实践的关系

### 14.1 已经对齐的部分

| 维度 | 已对齐的实践 |
|---|---|
| **LangGraph** | Graph 编排 · Interrupt/HITL · Persistence 思维 · State+Events 分层 |
| **MCP** | Tool-first · Streamable-HTTP · Client/Server 分离 |
| **主流 Agent 工程** | 模型可替换 · 工具契约化 · 长任务服务化 · 真实回归驱动 |

### 14.2 还没完全对齐到生产级的部分

| 差距 | 当前状态 | 生产级要求 |
|---|---|---|
| Durable checkpoint | `MemorySaver()` 内存级 | PostgreSQL / Redis 持久化 |
| 鉴权 | SSH Tunnel + localhost | OAuth / Token / API Key |
| Tracing | 本地测试脚本 + JSON 产物 | LangSmith / OpenTelemetry |
| Eval 体系 | 手动 case 测试 | 持续评测框架 |

这部分不是"做错了"，而是"还没做到下一阶段"。

---

## 15. 如果你要把这个项目当成学习样板，应该怎么读代码

### 推荐阅读路线

```mermaid
graph LR
    subgraph Step1["第1步 - 看入口"]
        F1["cli.py"]
        F2["agent_client.py"]
        F3["mcp_server.py"]
    end

    subgraph Step2["第2步 - 看连接"]
        F4["llm_factory.py"]
        F5["tool_adapter.py"]
        F6["tool_result_parser.py"]
    end

    subgraph Step3["第3步 - 看记忆"]
        F7["session_state.py"]
        F8["memory_store.py"]
        F9["context_builder.py"]
        F10["event_retriever.py"]
    end

    subgraph Step4["第4步 - 看工具"]
        F11["data_tools.py"]
        F12["combo_tools.py"]
        F13["train_tools.py"]
    end

    subgraph Step5["第5步 - 看服务"]
        F14["dataset_root.py"]
        F15["gpu_utils.py"]
        F16["train_service.py"]
        F17["train_log_parser.py"]
    end

    F1 --> F4
    F4 --> F7
    F7 --> F11
    F11 --> F14
```

### 第 1 步：看入口

先看 `cli.py` → `agent_client.py` → `mcp_server.py`

目标：理解用户输入从哪里进来，工具从哪里被注册出去

### 第 2 步：看连接

继续看 `llm_factory.py` → `tool_adapter.py` → `tool_result_parser.py`

目标：理解 provider abstraction、tool 调用结果如何被模型消费

### 第 3 步：看记忆

继续看 `session_state.py` → `memory_store.py` → `context_builder.py` → `event_retriever.py`

目标：理解 state 和 events 分层、prompt 不是固定文本而是动态上下文

### 第 4 步：看工具

继续看 `data_tools.py` → `combo_tools.py` → `train_tools.py`

目标：理解为什么有底层 tool 和高层 tool、tool contract 设计的重要性

### 第 5 步：看服务

继续看 `dataset_root.py` → `gpu_utils.py` → `train_service.py` → `train_log_parser.py`

目标：理解系统如何处理真实文件、真实 GPU、真实训练、真实重启恢复

---

## 16. 适合初学者记住的 15 个核心认知

1. Agent 不等于聊天机器人。
2. 模型不是执行器，而是规划器。
3. Tool 设计比 Tool 数量重要。
4. 复杂任务应该被压缩成高层组合工具。
5. 用户语言和程序参数之间经常需要翻译层。
6. 关键业务状态必须结构化。
7. 历史事件和当前状态应该分开存。
8. 不同模型 provider 之间不仅模型不同，消息协议细节也不同。
9. 高风险动作必须有人审兜底。
10. 资源策略应依赖运行时真实状态。
11. 长任务必须有独立 service 层。
12. 跨重启任务必须有 durable registry。
13. 日志不仅是文本，也可以是状态输入。
14. 真实数据和真实训练比漂亮 demo 更能暴露问题。
15. Agent 工程最终比拼的是"稳定可复用"，而不是"偶尔成功"。

---

## 17. 你可以自己动手做的学习实验

如果你想把这份项目当成跳板，我建议你自己做这些实验。

### 实验 1：切换 provider

目标：观察 DeepSeek 和 Ollama 在同一提示词下的行为差异

```bash
# Ollama
YOLOSTUDIO_LLM_PROVIDER=ollama python cli.py

# DeepSeek
YOLOSTUDIO_LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=xxx python cli.py
```

学习点：provider abstraction、模型规划差异、tool message 兼容问题

---

### 实验 2：删掉 SessionState 再试一次

目标：感受没有结构化状态时，多轮会话会多快失控

学习点：为什么不能只靠聊天记录

---

### 实验 3：把组合工具拆回底层工具

目标：对比 `prepare_dataset_for_training` 存在和不存在时复杂提示词的稳定性

学习点：为什么组合工具能降低对模型规划能力的依赖

---

### 实验 4：模拟 MCP 重启

目标：看 run registry 如何接管训练

学习点：durable state 的价值

---

### 实验 5：给一个非标准目录数据集

目标：看 dataset root resolver 如何表现

学习点：Agent 的问题很多其实来自"语义翻译失败"，而不是模型智商不够

---

## 18. 技术栈速查卡

```mermaid
graph TD
    ROOT["YoloStudio Agent"]

    ROOT --> PY["Python"]
    ROOT --> PROTO["Protocol"]
    ROOT --> LLM_G["LLM"]
    ROOT --> INFRA["Infrastructure"]
    ROOT --> BIZ["Business"]
    ROOT --> PERSIST["Persistence"]

    PY --> PY1["LangChain Core"]
    PY --> PY2["LangGraph"]
    PY --> PY3["langchain-mcp-adapters"]
    PY --> PY4["langchain-ollama"]
    PY --> PY5["langchain-openai"]

    PROTO --> PR1["MCP"]
    PROTO --> PR2["FastMCP"]
    PROTO --> PR3["Streamable HTTP"]

    LLM_G --> L1["Ollama"]
    LLM_G --> L2["DeepSeek"]
    LLM_G --> L3["OpenAI Compatible"]

    INFRA --> I1["SSH Tunnel"]
    INFRA --> I2["nvidia-smi"]
    INFRA --> I3["conda env"]
    INFRA --> I4["subprocess"]

    BIZ --> B1["YOLO / Ultralytics"]
    BIZ --> B2["DataHandler"]
    BIZ --> B3["YAML config"]

    PERSIST --> D1["JSON SessionState"]
    PERSIST --> D2["JSONL Events"]
    PERSIST --> D3["Run Registry"]
```

---

## 19. 术语表（给完全小白看的）

| 术语 | 简单解释 |
|---|---|
| **Agent** | 能理解任务、调用工具、完成流程的系统 |
| **Tool** | 模型可以调用的功能接口 |
| **MCP** | 把工具标准化暴露给模型的协议（Model Context Protocol） |
| **FastMCP** | MCP 协议的 Python 快速实现框架 |
| **ReAct** | 一边思考、一边调用工具、一边继续推理的模式 |
| **LangGraph** | 管理 Agent 状态流和工具调用顺序的编排框架 |
| **HITL** | Human-in-the-Loop，高风险动作由人确认 |
| **SessionState** | 当前会话的结构化状态 |
| **Event** | 会话里的历史动作记录 |
| **Context** | 每轮喂给模型的上下文组合 |
| **Provider** | 模型来源，例如 Ollama、DeepSeek |
| **Durable State** | 跨重启仍然存在的状态 |
| **Run Registry** | 记录长任务运行信息的持久化注册表 |
| **Readiness** | 训练前检查结果 |
| **Dataset Root** | 数据集根目录，而不是 images 子目录 |
| **Combo Tool** | 高层组合工具，用来减少模型拆步骤负担 |
| **SSH Tunnel** | 通过 SSH 加密隧道安全访问远程服务 |
| **Function Calling** | LLM 调用外部函数的标准能力 |
| **System Prompt** | 给模型设定角色和规则的指令文本 |
| **Tool Contract** | 工具的输入输出约定，包括参数、返回值、错误处理 |

---

## 20. 延伸阅读（官方资料）

这些链接能帮你把当前项目和更大的技术生态对应起来：

### Agent 框架
- LangGraph 官方文档
  <https://python.langchain.com/docs/langgraph>
- LangGraph Persistence
  <https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Interrupts
  <https://docs.langchain.com/oss/python/langgraph/interrupts>

### MCP 协议
- MCP 官方规范
  <https://modelcontextprotocol.io/>
- MCP Transports
  <https://modelcontextprotocol.io/specification/draft/basic/transports>
- MCP Authorization
  <https://modelcontextprotocol.io/specification/draft/basic/authorization>

### LLM Provider
- DeepSeek Function Calling
  <https://api-docs.deepseek.com/guides/function_calling>
- Ollama 文档
  <https://ollama.com/>

### GPU / 训练
- NVIDIA NVML / nvidia-smi 文档
  <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- Ultralytics YOLO 文档
  <https://docs.ultralytics.com/>

---

## 21. 最后一段总结

如果你把这个项目当成一个"Agent 学习样板"，它最有价值的地方不是某一个函数写得多漂亮，而是它完整展示了这一条链：

```text
自然语言
  -> 模型规划
  -> 工具调用
  -> 服务执行
  -> 状态回写
  -> 风险确认
  -> 长任务管理
  -> 重启恢复
  -> 真实训练闭环
```

```mermaid
graph LR
    A["自然语言输入"] --> B["模型规划 Planning"]
    B --> C["工具调用 Tool Call"]
    C --> D["服务执行 Execution"]
    D --> E["状态回写 State"]
    E --> F["风险确认 HITL"]
    F --> G["长任务管理"]
    G --> H["重启恢复 Registry"]
    H --> I["真实训练闭环"]

    style A fill:#E8F5E9
    style I fill:#E8F5E9
```

一旦这条链真的跑通，你对 Agent 的理解就会从：

- "大模型会不会回答"

变成：

- "一个 Agent 系统如何把模型、工具、状态、执行和风险控制拼成可工作的软件"

这正是这个项目最值得学习的地方。
