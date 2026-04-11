# YoloStudio Agent 项目新会话交接（精简版，2026-04-11）

> 用途：给新会话快速续上项目状态。  
> 这份文档刻意**少写本轮 SSH/连接细节**，重点只保留项目本身的阶段、代码状态、验证状态和下一步方向。

---

## 1. 一句话状态

截至 2026-04-11，`D:\yolodo2.0\agent_plan` 已经从“Agent 原型”推进到：

> **训练主链路已进入可实用状态；预测主链路已完成本地能力建设、本地回归与远端真实验证。**

---

## 2. 当前已经做出来的东西

### 2.1 数据准备 / 训练链路

已经完成并接入 Agent 的核心能力：

- `scan_dataset`
- `validate_dataset`
- `run_dataset_health_check`
- `detect_duplicate_images`
- `generate_yaml`
- `training_readiness`
- `prepare_dataset_for_training`
- `start_training`
- `check_training_status`
- `stop_training`
- `check_gpu_status`

这条链路已经补齐了：

- dataset root 解析
- 非标准目录容错
- 脏数据风险表达
- classes.txt 保留
- 训练参数来源标注
- MCP 重启后的 run registry / reattach
- file checkpoint / durable HITL 第一版

### 2.2 数据提取 / 预测链路

已经完成并接入 Agent 的核心能力：

- `preview_extract_images`
- `extract_images`
- `scan_videos`
- `extract_video_frames`
- `predict_images`
- `predict_videos`
- `summarize_prediction_results`

并且已经补齐：

- 图片抽取预览 / 真执行两段式
- 抽取结果可直接接 `scan_dataset / validate_dataset / prepare_dataset_for_training`
- prediction 结果 grounded reply
- prediction 路由
- prediction 别名兼容层
- prediction 会话状态写回
- 本地真实权重 / 真实视频测试链路
- 远端真实视频 prediction 回归基线

当前远端边界：
- `scan_videos`、图片抽取相关工具已同步可用
- `extract_video_frames` 远端仍依赖 `cv2 / numpy`，服务端现在会优雅报错，不再影响 MCP server 启动

---

## 3. 现在到底哪些是真的“完成了”

### 已经较稳的

#### 训练相关
- 数据准备 → 训练启动 → 状态查询 → 停止
- 脏数据集训练前准备
- 长任务训练生命周期观察
- MCP 重启后的训练接管

#### 数据治理相关
- 健康检查
- 重复检测
- 缺失标签风险表达
- `classes.txt` 类名保留

#### 预测相关（本地）
- 图片预测
- 图片目录批量预测
- 视频预测
- 视频目录批量预测
- 预测结果汇总

### 已经补齐的远端 prediction

- 远端环境自动识别（`yolodo` / `yolo`）
- 远端真实 prediction 执行
- 远端 prediction 回归基线
- 远端输出结果拉回本地归档

---

## 4. 代码结构现状

当前重点目录：

- `D:\yolodo2.0\agent_plan\agent\client`
- `D:\yolodo2.0\agent_plan\agent\server\tools`
- `D:\yolodo2.0\agent_plan\agent\server\services`
- `D:\yolodo2.0\agent_plan\agent\tests`
- `D:\yolodo2.0\agent_plan\deploy\scripts`
- `D:\yolodo2.0\agent_plan\doc`

### 当前关键文件

- `D:\yolodo2.0\agent_plan\agent\client\agent_client.py`
- `D:\yolodo2.0\agent_plan\agent\client\tool_adapter.py`
- `D:\yolodo2.0\agent_plan\agent\client\file_checkpointer.py`
- `D:\yolodo2.0\agent_plan\agent\server\tools\data_tools.py`
- `D:\yolodo2.0\agent_plan\agent\server\tools\train_tools.py`
- `D:\yolodo2.0\agent_plan\agent\server\tools\predict_tools.py`
- `D:\yolodo2.0\agent_plan\agent\server\tools\combo_tools.py`
- `D:\yolodo2.0\agent_plan\agent\server\services\train_service.py`
- `D:\yolodo2.0\agent_plan\agent\server\services\predict_service.py`
- `D:\yolodo2.0\agent_plan\agent\server\mcp_server.py`

### 目前结构上的真实问题

功能已经做出来了，但有几处文件开始变厚：

- `agent_client.py`
- `data_tools.py`
- `predict_service.py`

当前判断：

> **还不是彻底屎山，但继续扩功能前，应该安排一轮结构整理。**

---

## 5. 验证状态

### 训练主链路

这部分已经有较深验证：

- 本地单测 / 冒烟
- 脏数据压力测试
- 主线回归矩阵
- 长任务训练测试
- 远端真实训练验证

### 预测主链路

这部分当前主要是：

- 本地工具级测试
- 本地路由测试
- 本地 regression
- 本地真实素材验证
- 远端真实 prediction 验证

其中远端真实验证已确认：

- 输出归档：`agent/tests/test_prediction_remote_real_media_output.json`
- 真实结果：2 个视频、24 帧、13 个检测帧、15 个检测框
- 主要类别：`two_wheeler=15`

同时补了两个远端阻塞修复：

- `predict_service.py` 去掉对 `utils.label_writer` 的硬依赖
- 远端验证脚本按 `manifest.json` 固定权重选择，不再被上传时间戳误导

---

## 6. 当前最值得继续做的事

### 第一优先

做一轮 prediction 结构整理与回归固化：

1. 拆 `predict_service.py`
2. 统一本地 / 远端 prediction 验证入口
3. 把当前远端回归结果沉淀成更稳定的回归基线

### 第二优先

继续整理变厚文件：

1. 拆 `agent_client.py`
2. 拆 `data_tools.py`
3. 收口 prediction 相关脚本与测试资产组织

### 第三优先

再继续扩：

- 预测结果二次筛选/分析
- 批处理增强
- 更细的数据治理流程

---


## 6.1 补充：prediction 远端真实验证已经完成第一轮

在这份交接文档创建之后，项目又往前推进了一步：

- 已完成真实权重/视频上传
- 已同步 prediction 相关代码到远端
- 已在远端 `yolodo` conda 环境完成一轮真实视频 prediction
- 已拉回本地结果文件

关键结果：
- 处理视频：`2`
- 总帧数：`24`
- 有检测帧：`13`
- 总检测框：`15`
- 主要类别：`two_wheeler=15`

本地结果文件：
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media_output.json`

因此新会话不应再把 prediction 理解为“只有本地验证”，而应理解为：

> **prediction 已完成第一轮远端真实执行，下一步是把它收成固定回归基线，并继续补图片 prediction 的远端验证。**

## 7. 当前最重要的测试/文档文件

### 文档

- `D:\yolodo2.0\agent_plan\doc\project_summary.md`
- `D:\yolodo2.0\agent_plan\doc\current_progress_2026-04-11.md`
- `D:\yolodo2.0\agent_plan\doc\agent_test_playbook_2026-04-10.md`
- `D:\yolodo2.0\agent_plan\doc\prediction_regression_report_2026-04-11.md`
- `D:\yolodo2.0\agent_plan\doc\prediction_real_media_validation_2026-04-11.md`
- `D:\yolodo2.0\agent_plan\doc\prediction_remote_real_media_validation_2026-04-11.md`

### 测试

- `D:\yolodo2.0\agent_plan\agent\tests\test_mainline_regression_matrix.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_zyb_long_training_lifecycle.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_regression_suite.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_real_media_local_suite.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media.py`

---

## 8. 新会话推荐开场白

可直接复制下面这段：

```text
继续 D:\yolodo2.0\agent_plan 项目。

先读取：
D:\yolodo2.0\agent_plan\doc\new_conversation_handoff_2026-04-11.md
D:\yolodo2.0\agent_plan\doc\current_progress_2026-04-11.md
D:\yolodo2.0\agent_plan\doc\project_summary.md
D:\yolodo2.0\agent_plan\doc\agent_test_playbook_2026-04-10.md

然后直接继续推进，不用先问我。
当前优先做 prediction 这条线的结构整理，并继续固化远端真实验证基线。
```

---

## 9. 最后一句话

> **当前项目不是“没做出东西”，而是“训练链已经实用，预测链已经完成远端真实验证，接下来该做的是整理结构并把基线固化得更稳”。**


## 10. 最新阶段补充：训练知识库 Phase 1 已落地

当前新会话不应再把项目理解为“只有训练链 / prediction 链 / 数据提取链”，因为知识解释层已经开始工作。

已新增：

- `knowledge/core/`
- `knowledge/families/yolo/`
- `knowledge/playbooks/`
- `agent/server/services/knowledge_service.py`
- `agent/server/tools/knowledge_tools.py`

已接入工具：

- `retrieve_training_knowledge`
- `analyze_training_outcome`
- `recommend_next_training_step`

当前默认理解应为：

> 项目已经进入“工具执行层 + 知识解释层”并行阶段。

当前最自然的下一步，不再是继续大规模拆分厚文件，而是：

1. 扩首批训练规则覆盖面
2. 把训练结果汇总能力再补强，给知识分析更多事实输入
3. 再逐步为未来多模型族预留更清晰接口


## 11. 补充：知识库当前的真实性边界

当前知识库已增加来源隔离护栏：

- 默认只用 `official + workflow`
- `case` 默认不自动进入建议
- `test` 默认禁止进入建议

因此新会话不应把当前知识库理解为“已有真实训练经验库”，更准确的理解应为：

> **当前是规则库 / 流程知识库，真实训练案例层还没有开始大规模沉淀。**
