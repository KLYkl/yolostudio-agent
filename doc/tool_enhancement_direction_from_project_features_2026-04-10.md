# YoloStudio Agent 工具增强方向分析（基于现有项目功能）

> 目的：不是凭感觉决定“下一步增强哪个 tool”，而是先回看 `C:\workspace\yolodo2.0` 现有项目功能，再把 **GUI / core 已有能力** 映射到 **Agent 当前工具能力**，最后给出更合理的增强方向。

---

## 1. 结论先说

当前项目的功能面，实际上比 Agent 现在暴露出来的工具能力要大很多。

如果只看 `C:\workspace\yolodo2.0\core` 和 `C:\workspace\yolodo2.0\ui`，这个项目已经包含三大业务面：

1. **数据准备 / 数据治理**
2. **模型训练**
3. **预测推理 / 结果输出**

而当前 Agent 已经覆盖得比较好的，主要还是：

- scan
- validate
- split
- augment
- generate_yaml
- readiness
- prepare_dataset_for_training
- start/check/stop training
- GPU 状态

也就是说：

> **Agent 当前已经把“数据准备到训练”这条主线做厚了，但对“数据治理深水区”和“预测推理主线”的覆盖还远远不够。**

所以，下一步工具增强不应该只靠脑补，而应按下面这个原则：

### 原则
1. **优先增强已经进入主线的工具**（收益最大）
2. **再把 core 中“纯 Python / 低耦合 / headless 友好”的能力接进 Agent**
3. **最后才考虑强 Qt 耦合、强实时交互、强 GUI 生命周期的功能**

---

## 2. 我这次看了哪些项目功能

### 2.1 程序总入口与主界面
我看了：

- `C:\workspace\yolodo2.0\main.py`
- `C:\workspace\yolodo2.0\ui\main_window.py`

结论：
主程序分成 3 个主 Tab：

1. **数据准备**
2. **模型训练**
3. **预测推理**

这和我们现在 Agent 主线的划分其实是高度一致的，只是 Agent 目前只把前两块做得比较深。

---

### 2.2 数据准备模块
我看了：

- `C:\workspace\yolodo2.0\ui\data_widget\_widget.py`
- `C:\workspace\yolodo2.0\core\data_handler\_handler.py`
- 以及相关 mixin：
  - `_scan.py`
  - `_validate.py`
  - `_split.py`
  - `_augment.py`
  - `_extract.py`
  - `_image_check.py`
  - `_video_extract.py`
  - `_modify.py`
  - `_convert.py`
  - `_stats.py`

结论：
DataHandler 其实非常大，能力不只是我们现在已经 Agent 化的这些。

### DataHandler 已有能力清单

#### A. 已经部分 Agent 化的
- `scan_dataset`
- `validate_labels`
- `split_dataset`
- `generate_yaml`
- `augment_dataset`

#### B. 还没 Agent 化但非常有价值的
- `run_health_check`
- `check_image_integrity`
- `analyze_image_sizes`
- `detect_duplicates`
- `export_check_report`
- `extract_images`
- `preview_extract`
- `extract_video_frames`
- `scan_videos`
- `convert_format`
- `preview_convert_format`
- `modify_labels`
- `preview_modify_labels`
- `clean_orphan_labels`
- `generate_missing_labels`
- `generate_empty_labels`
- `categorize_by_class`

这说明：

> **当前 Agent 在数据治理这块，还只动用了 DataHandler 的一部分。真正的“数据治理型 Agent”能力，还有很大空间。**

---

### 2.3 模型训练模块
我看了：

- `C:\workspace\yolodo2.0\ui\train_widget.py`
- `C:\workspace\yolodo2.0\core\train_handler.py`

结论：
训练 UI 的功能面其实也比当前 Agent 暴露的更宽。

#### 当前训练 UI 已体现的能力
- Python/conda 环境扫描
- 模型文件选择
- 数据 YAML 选择
- 基础超参数配置
- 高级超参数配置
- 命令预览
- 实时日志查看
- 启动 / 停止训练

#### 当前 Agent 已覆盖的
- `start_training`
- `check_training_status`
- `stop_training`
- GPU 状态与 auto device 解析

#### 当前 Agent 还没覆盖但有潜力增强的
- 训练前命令/参数预览
- 更完整的高级超参控制
- 更清晰的训练 run identity
- 更细的日志/metrics 摘要
- 训练历史/实验记录

所以训练工具下一步增强，不是“能不能启动”，而是：

> **能不能更像实验控制接口，而不是一个薄薄的启动壳。**

---

### 2.4 预测推理模块
我看了：

- `C:\workspace\yolodo2.0\ui\predict_widget\_widget.py`
- `C:\workspace\yolodo2.0\core\predict_handler\_manager.py`
- `C:\workspace\yolodo2.0\core\predict_handler\_image_batch.py`
- `C:\workspace\yolodo2.0\core\predict_handler\_video_batch.py`
- `C:\workspace\yolodo2.0\core\output_manager.py`
- `C:\workspace\yolodo2.0\core\camera_scanner.py`

结论：
预测模块其实是当前项目里**功能最宽但 Agent 覆盖最少**的一块。

#### 现有项目的预测能力
- 单图预测
- 图片目录批量预测
- 单视频预测
- 视频批量预测
- 摄像头扫描与摄像头输入
- 屏幕扫描与屏幕输入
- RTSP 测试与 RTSP 输入
- 输出目录管理
- 关键帧保存
- 视频保存
- 报告生成
- 路径列表导出

#### 当前 Agent 基本没覆盖
当前 Agent 在预测这块几乎还没正式暴露主线工具。

所以这部分以后一定会成为：

> **Agent 第二主线最自然的扩展方向。**

---

## 3. 当前 Agent 工具和项目原功能的映射

### 3.1 当前 MCP 工具
我看了：

- `C:\workspace\yolodo2.0\agent_plan\agent\server\mcp_server.py`

当前工具有：
- `scan_dataset`
- `split_dataset`
- `validate_dataset`
- `augment_dataset`
- `generate_yaml`
- `training_readiness`
- `prepare_dataset_for_training`
- `start_training`
- `check_training_status`
- `stop_training`
- `check_gpu_status`

### 3.2 映射结果

| 项目原能力 | 当前 Agent 覆盖情况 | 结论 |
|---|---|---|
| 数据扫描 | 已覆盖 | 但仍可增强 |
| 标签校验 | 已覆盖 | 但还不够“治理化” |
| 数据划分 | 已覆盖 | 主线已成型 |
| YAML 生成 | 已覆盖 | 主线已成型 |
| 数据增强 | 已覆盖 | 但仍偏基础 |
| 训练启动/停止/状态 | 已覆盖 | 已是主线核心 |
| GPU 状态 | 已覆盖 | 已较成熟 |
| 图像健康检查 | 未覆盖 | 高价值，建议尽快接 |
| 重复检测 | 未覆盖 | 高价值，建议接 |
| 图片抽取 | 未覆盖 | 中高价值 |
| 视频抽帧 | 未覆盖 | 中高价值 |
| 标签修改/清理 | 未覆盖 | 高风险，建议后置 |
| 格式转换 | 未覆盖 | 中等价值 |
| 单图预测 | 未覆盖 | 高价值，建议作为第二主线起点 |
| 图片批量预测 | 未覆盖 | 高价值 |
| 视频批量预测 | 未覆盖 | 高价值但复杂 |
| 摄像头/屏幕/RTSP | 未覆盖 | 价值高，但更偏实时系统，不建议最先做 |
| 报告导出 | 部分覆盖 | 可继续增强 |

---

## 4. 所以“增强方向”应该怎么定

不是所有现有项目功能都该一股脑 Agent 化。
应该按 **价值 / 耦合度 / 主线关系 / 风险** 四个维度来排。

---

## 5. 第一优先级：先增强现有核心工具

这部分最值钱，因为已经在主线里。

### 5.1 `scan_dataset`
建议增强方向：
- 更细的类别分布摘要
- 图像尺寸分布摘要
- 目录结构解释更完整
- classes.txt / data.yaml 检测再稳一点
- 更多“next_actions”建议

### 为什么先增强它
因为它是几乎所有后续动作的入口。
`scan` 强，后面的 `readiness / prepare / train` 都会更稳。

---

### 5.2 `validate_dataset`
建议增强方向：
- 把 issue 分级（fatal / warning / info）
- 给出更明确的修复建议
- 增加 issue samples
- 把图像质量问题和标签问题更清晰分开

### 为什么先增强它
因为当前它还偏“检测器”，还不够像“治理建议器”。

---

### 5.3 `training_readiness`
建议增强方向：
- 成为统一训练前检查入口
- 更明确区分：
  - 技术上可训练
  - 数据质量上不建议训练
- 统一输出：
  - blockers
  - warnings
  - recommendations
  - recommended_start_training_args

### 为什么先增强它
它是主线里最该继续升格的工具。

---

### 5.4 `prepare_dataset_for_training`
建议增强方向：
- dry-run 模式
- 更明确列出将创建哪些产物
- nonstandard / unknown 目录更好的恢复建议
- 更稳定地复用 classes.txt
- 更清楚地解释为什么 split / 为什么不 split

### 为什么先增强它
因为它已经是“复杂训练提示词”的核心落点。

---

### 5.5 `start_training`
建议增强方向：
- 参数来源标记更完整
- dry-run / preflight-only 模式
- 更清楚的 run_id / output_dir / log_file
- 更适合后续实验追踪

### 为什么先增强它
当前训练工具的下一阶段，不是“能起就行”，而是“更像实验控制接口”。

---

## 6. 第二优先级：把 DataHandler 里低耦合高价值能力接进来

这部分是“增强原有工具”之后最值得做的。

### 6.1 图像健康检查工具（强烈建议）
建议新增：
- `run_dataset_health_check`
- `export_dataset_health_report`

对应原能力：
- `run_health_check`
- `export_check_report`

### 解决什么问题
把现在偏“扫描/标签校验”的能力，扩展成真正的：

> **数据质量治理工具**

### 为什么值钱
因为现实数据集的问题，很多不只是标签错误，还有：
- 图片损坏
- 尺寸异常
- 重复图片
- 零字节文件

这些当前 Agent 主线还没有系统覆盖。

---

### 6.2 重复检测工具（强烈建议）
建议新增：
- `detect_duplicate_images`

对应原能力：
- `detect_duplicates`

### 为什么值钱
这是脏数据集里非常常见的真实问题，而且非常适合 agent 汇报和建议。

---

### 6.3 图片抽取工具
建议新增：
- `preview_extract_images`
- `extract_images`

对应原能力：
- `preview_extract`
- `extract_images`

### 为什么值钱
适合这类自然语言：
- “从这个大目录里抽一些样本”
- “按类别各抽 100 张”
- “从每个子目录抽固定数量”

这类任务天然适合 Agent。

---

### 6.4 视频抽帧工具
建议新增：
- `scan_videos`
- `extract_video_frames`

对应原能力：
- `scan_videos`
- `extract_video_frames`

### 为什么值钱
这和当前数据准备主线是自然衔接的，尤其适合：
- 视频转训练集
- 从监控视频中抽样

---

## 7. 第三优先级：开启第二主线——预测 / 批处理推理

这部分不是当前主线的补丁，而是下一条业务线。

### 7.1 最推荐先做的预测工具
建议顺序：

#### A. `predict_images`
覆盖：
- 单图
- 图片目录批量

#### B. `predict_videos`
覆盖：
- 单视频
- 视频目录批量

#### C. `summarize_prediction_results`
把输出做成适合 Agent 总结的结果。

### 为什么先做这三类
因为它们最接近：
- `ImageBatchProcessor`
- `VideoBatchProcessor`
- `OutputManager`

并且是 headless 化最自然的一批。

---

## 8. 暂时不建议最先 Agent 化的功能

### 8.1 摄像头 / 屏幕 / RTSP 实时预测
为什么先不优先：
- 更强实时性
- 更强状态机
- 更强 GUI / runtime 耦合
- 更难测试与回放

不是不能做，而是：

> **现在做它的收益，不如先做 image/video batch 预测。**

---

### 8.2 标签批量修改 / 清理
为什么后置：
- 破坏性更强
- 风险更高
- 更需要 preview + diff + 审批

如果以后做，应该是：
- preview first
- HITL strong
- possibly backup enforced

---

### 8.3 格式转换
为什么不是第一优先：
- 有价值
- 但对当前主线和第二主线的直接增益，不如健康检查 / 抽取 / 预测

---

## 9. 我给你的工具增强路线图

### 阶段 A：继续做厚主线核心工具
优先顺序：
1. `training_readiness`
2. `prepare_dataset_for_training`
3. `start_training`
4. `scan_dataset`
5. `validate_dataset`

### 阶段 B：引入数据治理能力
优先顺序：
1. `run_dataset_health_check`
2. `detect_duplicate_images`
3. `export_dataset_health_report`
4. `preview_extract_images`
5. `extract_images`
6. `extract_video_frames`

### 阶段 C：开启第二主线（预测）
优先顺序：
1. `predict_images`
2. `predict_videos`
3. `summarize_prediction_results`
4. `export_prediction_report`

---

## 10. 一句话判断

> **如果完全对照项目原有功能来看，当前 Agent 最值得增强的方向不是“继续围着训练打转”，而是：先把已有主线工具做厚，再把 DataHandler 里的健康检查/抽取能力接进来，最后再正式打开预测/批处理推理这条第二主线。**

这条路和项目原有功能版图是最一致的，也最符合“先低耦合高价值，再逐步扩面”的演进方式。

## 已落地的第一批 Agent 化增强（2026-04-10）

当前已按上述原则落地两项新工具：

1. `run_dataset_health_check`
   - 解决问题：图片损坏、零字节、格式不匹配、尺寸异常、重复图片等问题，过去缺少统一的只读检查入口。
   - Agent 适配点：
     - 输入使用 `dataset_path`，支持 dataset root
     - 输出提供 `summary / warnings / risk_level / issue_count / next_actions`
     - 可选 `export_report`，但默认只读不改数据

2. `detect_duplicate_images`
   - 解决问题：过去重复图片能力只存在桌面逻辑里，Agent 没有稳定的结构化入口。
   - Agent 适配点：
     - 输出 `duplicate_groups / duplicate_extra_files / groups`
     - `groups` 提供 sample paths，方便 grounded 回答
     - 默认只做检测，不直接清理

这一批增强说明：
- 不是把桌面功能原样暴露，而是先做成更适合自然语言调用、结构化输出、可测试、可追踪的 Agent 工具。
- 后续同类能力（如健康报告、抽图、抽帧、预测批处理）都应沿用同样的 Agent 适配原则。
