# YoloStudio Agent 会话续接文档

> 用途：给未来的新会话或中断后的续做提供一个“先读这一份就能接上”的入口。
> 这不是完整项目说明书，而是**当前协作上下文的压缩版交接文档**。
> 更新时间：2026-04-11

---

## 1. 一句话结论

当前项目已经进入：

> **第一主线（数据准备 -> 训练）基本进入实用态；第二主线（图片 / 图片目录 / 视频 / 视频目录预测 + 结果汇总）已完成本地工具级落地，下一步应优先做远端部署与真实验证。**

更直白一点：

- 第一主线已经不是 demo 了
- 主线训练、脏数据、恢复接管、checkpoint、测试手册都已经成型
- 第二主线已经起线，并补到了**预测结果汇总与视频预测**
- 目前还是**本地验证为主**
- 下一步最有价值的事情，不是再发散加功能，而是把第二主线真正落到远端环境并验证

---

## 2. 如果是新会话，先读哪些文件

优先顺序：

1. `D:\yolodo2.0\agent_plan\doc\conversation_context_handoff.md`（本文件）
2. `D:\yolodo2.0\agent_plan\doc\current_progress_2026-04-11.md`
3. `D:\yolodo2.0\agent_plan\doc\project_summary.md`
4. `D:\yolodo2.0\agent_plan\doc\agent_test_playbook_2026-04-10.md`
5. `D:\yolodo2.0\agent_plan\doc\mainline_regression_matrix_report_2026-04-11.md`
6. `D:\yolodo2.0\agent_plan\doc\prediction_regression_report_2026-04-11.md`

如果只想最快进入状态，至少读前 3 个。

---

## 3. 当前阶段判断

### 第一主线
第一主线指：

> 数据集进入系统 -> 数据准备 -> 训练前判断 -> 启动训练 -> 查询 / 停止 -> 恢复接管

当前判断：
- 已达到 **单人、内网、有人值守** 的稳定可用门槛
- 当前剩余问题，已经不是“这条链能不能跑”，而是更偏工程化收尾

### 第二主线
第二主线指：

> 图片 / 图片目录 / 视频 / 视频目录预测 -> 预测结果汇总 -> grounded 结果总结 ->（后续）RTSP / 摄像头 / 屏幕流

当前判断：
- 已完成 **Phase 2：headless 图片 / 图片目录 / 视频 / 视频目录预测 + 结果汇总**
- 已完成本地工具级验证
- **还没完成远端部署与远端真实验证**

---

## 4. 最近一个关键里程碑

最近一个明确里程碑提交是：

- 仓库：`D:\yolodo2.0\agent_plan`
- commit：`f712b7b`
- message：`feat: add durable checkpoints and image prediction mainline`

这个提交的实际意义：

### 4.1 第一主线补上 durable checkpoint
- 新增：`D:\yolodo2.0\agent_plan\agent\client\file_checkpointer.py`
- 作用：LangGraph 的 interrupt / HITL 不再只依赖进程内内存
- 当前 checkpoint 文件落在：
  - `D:\yolodo2.0\agent_plan\memory\checkpoints\<session>.pkl`

### 4.2 第二主线正式起线
- 新增：`D:\yolodo2.0\agent_plan\agent\server\services\predict_service.py`
- 新增：`D:\yolodo2.0\agent_plan\agent\server\tools\predict_tools.py`
- MCP 新工具：`predict_images`
- MCP 新工具：`summarize_prediction_results`
- MCP 新工具：`predict_videos`
- Agent 侧已接入：
  - 预测意图路由
  - 预测 grounded reply
  - `active_prediction` 会话状态
  - 预测结果汇总路由与 grounded 汇总

### 4.3 第二主线已有本地验证基线
- `D:\yolodo2.0\agent_plan\agent\tests\test_predict_tools.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_route.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_regression_suite.py`
- 报告：`D:\yolodo2.0\agent_plan\doc\prediction_regression_report_2026-04-11.md`
- 当前本地预测回归分数：`1.0`
- 追加完成：
  - `summarize_prediction_results` 的工具级验证
  - `predict_videos` 的工具级验证
  - 旧汇总工具名/参数名兼容层

> 注意：本地存在 `asyncio/_overlapped` 环境异常，会阻塞一部分依赖 LangChain 的回归脚本执行。
> 因此当前第二主线最可信的最新验证依据是：
> - `py_compile`
> - `test_predict_tools.py`
> - 代码级 route / alias / grounded 集成检查

---

## 5. 当前实际已完成的能力

### 5.1 第一主线：数据准备到训练
已经具备：
- dataset root 解析
- 非标准目录别名识别（如 `pics/ann`）
- `scan_dataset`
- `validate_dataset`
- `split_dataset`
- `generate_yaml`
- `training_readiness`
- `prepare_dataset_for_training`
- `start_training`
- `check_training_status`
- `stop_training`
- MCP 重启后 run registry / 训练接管
- durable checkpoint / persistent HITL（本地文件级）

### 5.2 数据质量治理
已经具备：
- `run_dataset_health_check`
- `detect_duplicate_images`
- classes.txt 语义保留
- 缺失标签风险表达
- dirty dataset（`zyb`）长任务训练验证

### 5.3 第二主线：预测 Phase 2
已经具备：
- 单张图片预测
- 图片目录批量预测
- 单个视频预测
- 视频目录批量预测
- 标注图输出
- YOLO 标签输出
- JSON 结果报告
- 视频预测报告
- `prediction_report.json` 汇总
- grounded 预测总结
- 旧工具名兼容：
  - `predict_directory`
  - `batch_predict_images`
  - `predict_images_in_dir`
- 旧参数名兼容：
  - `path/source/input_path/dir_path/folder -> source_path`
- 旧汇总工具名兼容：
  - `summarize_predictions`
  - `summarize_prediction_report`
  - `analyze_prediction_report`
- 旧汇总参数名兼容：
  - `path/report/json_report/file -> report_path`
  - `dir_path/folder/output -> output_dir`

---

## 6. 当前还没完成的事

### 6.1 第二主线最重要的未完成项
> **把 prediction 这条线真正部署到远端，并做真实图片预测验证。**

这个是下一步的第一优先级。

还没做的包括：
- 远端同步 `predict_service.py` / `predict_tools.py`
- 重启远端 MCP，确认 `predict_images` 已注册可用
- 用真实图片 / 图片目录 / 视频 / 视频目录做远端 prediction 验证
- 检查输出目录、标注图、JSON 报告是否正常
- 做第二主线的远端回归基线

### 6.2 第二主线后续扩展项
在远端 prediction 验证稳定后，再继续：
- 更复杂的筛选 / 导出工作流

### 6.3 第一主线剩余尾巴
虽然第一主线已接近稳定，但仍有几个小尾巴：
- 少量错误语义 / fail-fast 文案还可以继续收紧
- grounded reply 还能再扩大和再严格一点
- durable checkpoint 当前只是**单人、本地文件级**，不是共享服务级数据库方案

---

## 7. 当前最重要的测试资产

### 7.1 第一主线
- `D:\yolodo2.0\agent_plan\agent\tests\test_mainline_regression_matrix.py`
- `D:\yolodo2.0\agent_plan\doc\mainline_regression_matrix_report_2026-04-11.md`
- `D:\yolodo2.0\agent_plan\agent\tests\test_zyb_long_training_lifecycle.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_file_checkpointer.py`

### 7.2 第二主线
- `D:\yolodo2.0\agent_plan\agent\tests\test_predict_tools.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_predict_video_tools.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_route.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_regression_suite.py`
- `D:\yolodo2.0\agent_plan\doc\prediction_regression_report_2026-04-11.md`
- `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_real_media_local_suite.py`
- `D:\yolodo2.0\agent_plan\doc\prediction_real_media_validation_2026-04-11.md`

### 7.3 测试手册
- `D:\yolodo2.0\agent_plan\doc\agent_test_playbook_2026-04-10.md`

当前手册已经覆盖：
- 主线回归矩阵
- 训练轮数建议
- 持久化 checkpoint / HITL 测试
- 第二主线预测回归
- split 产物清理流程

---

## 8. 当前用到的关键数据集

### 8.1 第一主线标准训练/准备集
- `/home/kly/test_dataset`

用途：
- 标准数据准备到训练链路
- 复杂训练意图回归

### 8.2 第一主线脏数据/长任务集
- `/home/kly/agent_cap_tests/zyb`

用途：
- 脏数据风险表达
- health / duplicate 检查
- 长任务训练生命周期测试

### 8.3 非标准目录测试集
- `/home/kly/agent_cap_tests/nonstandard_dataset`

用途：
- `pics/ann` 这类非标准目录结构容错测试

### 8.4 真正未知结构测试集
- `/home/kly/agent_cap_tests/unknown_dataset`

用途：
- fail-fast / resolve_root 提前失败测试

### 8.5 本地大数据集
- `H:\fuyangben\zyb`

用途：
- 本地 health / duplicate 只读压力测试

### 8.6 本地真实预测素材
- 权重池：`C:\Users\29615\OneDrive\桌面\yuntian`
- 视频池：`H:\foto`

用途：
- 第二主线真实本地权重 / 视频验证
- 本机预测环境探测
- 真实素材 Mock 链路验证

---

## 9. 当前协作约定（重要）

用户已经明确授权：

> **默认持续推进，不为低价值事项停下来征求意见。**

所以后续工作方式应保持：
- 主线优先
- 低风险默认直接做
- 测试 / 文档 / git 提交一起完成
- 只有在真正高风险、不可逆、或业务方向有歧义时才停下来

### 需要遵守的工程边界
- 优先改 `D:\yolodo2.0\agent_plan\...`
- 不随意改 `D:\yolodo2.0\core\`、`ui\` 等主项目核心源码
- 文档 / 代码默认 UTF-8（无 BOM）
- 不把无关未提交文件混入主线提交

---

## 10. 如果下一会话要继续，推荐直接这样开场

推荐直接说：

```text
继续 D:\yolodo2.0\agent_plan 项目。
先读取：
D:\yolodo2.0\agent_plan\doc\conversation_context_handoff.md
D:\yolodo2.0\agent_plan\doc\current_progress_2026-04-11.md
D:\yolodo2.0\agent_plan\doc\project_summary.md
D:\yolodo2.0\agent_plan\doc\agent_test_playbook_2026-04-10.md
然后按主线继续推进，不用先问我。
```

---

## 11. 下一步的明确执行建议

如果不再征询用户，最合理的下一步顺序是：

1. **处理第二主线真实推理环境问题或切到远端预测环境**
   - 当前本机真实素材接入已验证
   - 但 `ultralytics / torch` 导入被 `WinError 10106` 阻塞
2. **远端部署 prediction Phase 2**
   - 同步 `predict_service.py` / `predict_tools.py`
   - 重启远端 MCP
3. **做远端真实预测验证**
   - 单图
   - 图片目录
   - 单视频
   - 视频目录
   - 输出工件
   - grounded 回答
4. **把第二主线也建立成远端回归基线**
5. **然后继续做 prediction 结果汇总 / 批处理增强**

---

## 12. 一句话收尾

> 如果只记住一件事：**第一主线已经基本收稳，第二主线已经在本地起线成功；下一步最值钱的事，是把第二主线真正部署到远端并做真实验证。**
