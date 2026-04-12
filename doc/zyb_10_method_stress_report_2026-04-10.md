# zyb 大数据集 10 方法全方位测试报告（2026-04-10）

> 数据集：`/data/agent_cap_tests/zyb`
>
> 说明：本轮测试的话术为**自拟**，不是随意聊天，而是按真实使用场景去构造：
> - 数据集扫描与校验
> - readiness 判断
> - prepare → train 主链路
> - 高风险确认链
> - “只判断不训练”约束
> - 训练取消后的参数回忆
> - Gemma / DeepSeek 双 provider 对照

---

## 1. 测试目的

这轮测试不是为了证明“系统没问题”，而是为了回答两个更实际的问题：

1. 当前版本在 **大数据量 + 脏数据** 场景下，主链路还能不能稳定工作？
2. 当前版本的**能力范围和边界**到底在哪里？

这里的“脏数据”指的是这份 `zyb` 数据具备以下特征：
- 图片很多：`7027` 张
- 缺失标签很多：`5179` 张
- `labels/` 里混有 `classes.txt`
- 已有类别语义：`Excavator / bulldozer / piling_machine / two_wheeler`

---

## 2. 本轮采用的 10 种测试方法

| # | 方法 | 类型 | 目标 |
|---|---|---|---|
| 1 | `tool_scan_root` | Tool 直调 | 看 root 识别、类名发现、缺失标签风险是否正确暴露 |
| 2 | `tool_validate_root` | Tool 直调 | 看校验阶段是否把大量缺失标签提升为风险 |
| 3 | `tool_training_readiness` | Tool 直调 | 看 readiness 是否同时表达 blocker 与 dirty dataset 风险 |
| 4 | `tool_prepare_dataset` | Tool 直调 | 看 prepare 是否保留真实类名并给出风险说明 |
| 5 | `tool_training_lifecycle` | Tool 直调 | 看 start/status/stop 真实训练生命周期是否稳定 |
| 6 | `gemma_dirty_summary` | Agent / Gemma | 看 Gemma 在脏数据摘要场景下是否能基于工具事实说话 |
| 7 | `gemma_no_train_constraint` | Agent / Gemma | 看“只判断不训练”约束是否能被遵守 |
| 8 | `gemma_full_chain_train` | Agent / Gemma | 看复杂自然语言能否稳定落回 prepare → train 两段式流程 |
| 9 | `gemma_cancel_and_recall` | Agent / Gemma | 看取消训练后，参数回忆与风险回忆是否准确 |
| 10 | `deepseek_full_chain_train` | Agent / DeepSeek | 看 DeepSeek 在同一复杂链路上的表现与 Gemma 对照 |

原始结果：`C:\workspace\yolodo2.0\agent_plan\agent\tests\test_zyb_10_method_output.json`
测试脚本：`C:\workspace\yolodo2.0\agent_plan\agent\tests\test_zyb_10_method_suite.py`

---

## 3. 逐项结果

### 方法 1：`tool_scan_root`
**输入**
- `scan_dataset('/data/agent_cap_tests/zyb')`

**目的**
- 验证 root 解析
- 验证 `classes.txt` 自动发现
- 验证缺失标签风险能否前移暴露

**结果**
- 成功识别：
  - `dataset_root=/data/agent_cap_tests/zyb`
  - `resolved_img_dir=/data/agent_cap_tests/zyb/images`
  - `resolved_label_dir=/data/agent_cap_tests/zyb/labels`
- 成功发现：
  - `detected_classes_txt=/data/agent_cap_tests/zyb/labels/classes.txt`
  - `class_name_source=classes_txt`
- 成功返回真实类名：
  - `Excavator`
  - `bulldozer`
  - `piling_machine`
  - `two_wheeler`
- 成功暴露数据风险：
  - `missing_label_images=5179`
  - `missing_label_ratio=0.737`
  - `risk_level=critical`

**结论**
- 这一步现在已经很稳。
- 说明主线这轮新增的 `classes.txt` 保留和风险表达，已经真正生效。

---

### 方法 2：`tool_validate_root`
**输入**
- `validate_dataset('/data/agent_cap_tests/zyb')`

**目的**
- 看 `validate_dataset` 还会不会只盯着“标签文件语法”，忽略大量缺失标签图片

**结果**
- `has_issues=false`
- `has_risks=true`
- `risk_level=critical`
- `summary` 已明确出现：
  - “发现 5179 张图片缺少标签（占比 73.7%）...”

**结论**
- 这一步相比之前已经明显提升。
- 现在 validate 不再只是“语法层没问题”，而是开始表达**训练风险**。

---

### 方法 3：`tool_training_readiness`
**输入**
- `training_readiness('/data/agent_cap_tests/zyb')`

**目的**
- 看 readiness 会不会同时表达：
  - hard blocker
  - dirty dataset 风险

**结果**
- `ready=false`
- `blockers=['缺少可用的 data_yaml']`
- 同时保留：
  - `warnings=['发现 5179 张图片缺少标签（占比 73.7%）...']`
  - `risk_level=critical`
  - `detected_classes_txt=.../classes.txt`

**结论**
- 现在 readiness 的表达已经比以前更像工程系统：
  - 既告诉你“当前为什么不能直接训”
  - 也告诉你“即使补齐 yaml，数据本身还存在什么风险”

---

### 方法 4：`tool_prepare_dataset`
**输入**
- `prepare_dataset_for_training('/data/agent_cap_tests/zyb')`

**目的**
- 看 prepare 主线是否已经真正把：
  - root
  - split
  - generate_yaml
  - readiness
  - 风险表达
  串成稳定闭环

**结果**
- `ready=true`
- 生成：
  - `/data/agent_cap_tests/zyb/images_split/data.yaml`
- `class_name_source=classes_txt`
- `summary` 明确写出：
  - “数据集已准备到可训练状态，但存在数据质量风险 ...”

**结论**
- prepare 主线已经可用，而且比之前更可信：
  - 不只会“把事情做完”
  - 还会把 dirty dataset 风险留在结果里

---

### 方法 5：`tool_training_lifecycle`
**输入**
- 先 `prepare_dataset_for_training`
- 再 `start_training(model=/models/yolov8n.pt, epochs=3, device=auto)`
- 再 `check_training_status`
- 再 `stop_training`

**目的**
- 验证真实训练生命周期是否稳定

**结果**
- 训练成功启动
- 8 秒后查询：
  - `running=true`
  - `device=1`
- `stop_training` 成功
- 最后状态：
  - `running=false`

**结论**
- 当前主线不只是“能准备数据”，还能够真实地发起和管理训练。
- 从工程角度看，这条链已经能支持真实使用。

---

### 方法 6：`gemma_dirty_summary`
**话术**
- `请扫描 /data/agent_cap_tests/zyb/ ，然后用中文总结这个数据集目前最值得注意的 3 个点。`

**目的**
- 看 Gemma 在 dirty dataset 摘要任务里，能否老老实实依据工具事实作答

**结果**
- 它完成了回答，但问题明显：
  - 没准确提到 `5179`
  - 没提到真实类名
  - 也没提到 `classes.txt`
  - 给出的数字明显是“估计式”而不是工具结果复述

**结论**
- **Gemma 的执行链路强于解释链路。**
- 它能用工具做事，但在“开放式总结”里仍然会把事实说虚。

---

### 方法 7：`gemma_no_train_constraint`
**话术**
- `请检查 /data/agent_cap_tests/zyb/ 是否能直接训练。如果还不能，请只告诉我原因和建议，不要启动训练，也不要先做划分。`

**目的**
- 看“不要启动训练”这种约束是否能被遵守

**结果**
- 它确实**没有**启动训练，也**没有**先做 prepare/split
- 但解释里仍然存在问题：
  - 类别被说成了并不存在的 `car / person / bicycle / motorcycle`
  - 计数也出现漂移

**结论**
- 约束层是稳的：不会乱开训练
- 解释层仍然不稳：会把分析说跑偏

---

### 方法 8：`gemma_full_chain_train`
**话术**
- `数据在 /data/agent_cap_tests/zyb/，按默认划分比例，然后用 yolov8n 模型进行训练。`

**目的**
- 验证 Gemma 在复杂多步自然语言下，能不能稳定走主线

**结果**
- 稳定进入两段式流程：
  1. `prepare_dataset_for_training(force_split=true)`
  2. `start_training(...)`
- 二次确认后训练真实启动
- 后续 `check_training_status / stop_training` 都成功

**结论**
- 这说明：
  - 当前主线对 Gemma 已经足够“收口”
  - 它虽然解释层一般，但执行主链路已经可用

---

### 方法 9：`gemma_cancel_and_recall`
**话术**
1. `数据在 /data/agent_cap_tests/zyb/，按默认划分比例，然后用 yolov8n 模型进行训练。`
2. 在第二次确认处取消训练
3. 再问：`刚才待确认的训练参数是什么？这个数据集最大的风险又是什么？`

**目的**
- 看取消后的参数回忆和风险回忆是否准确

**结果**
- 它能回忆出“这是一次 yolov8n 训练”
- 也知道“最大风险是大量缺失标签”
- 但仍有严重漂移：
  - 把 `data_yaml` 编造成了 `/home/user/agent_data/images_split/data.yaml`
  - 把 `epochs` 说成了 `100`
  - 没保留“这是刚才待确认而未执行的参数”那种精确边界

**结论**
- 当前 `SessionState / Memory` 已经让它“知道大概发生了什么”
- 但 **Gemma 对参数精确回忆仍然不可靠**

---

### 方法 10：`deepseek_full_chain_train`
**话术**
- `数据在 /data/agent_cap_tests/zyb/，按默认划分比例，然后用 yolov8n 模型进行训练。`

**目的**
- 对照 DeepSeek 在同样复杂链路上的表现

**结果**
- 同样稳定走通：
  1. `prepare_dataset_for_training(force_split=true)`
  2. `start_training(...)`
- 训练真实启动
- 状态可查
- stop 成功
- 而且在最终解释里它能明确写出：
  - 4 个真实类别名
  - 5179 张缺失标签
  - 73.7% 风险
  - `GPU 1（auto 解析）`

**结论**
- DeepSeek 在这类“复杂链路 + 解释总结”场景下，明显比 Gemma 更稳。

---

## 4. 综合判断

### 4.1 当前已经明显稳住的能力
以下范围我认为已经比较可靠：

1. **大数据集 root 识别**
2. **dirty dataset 的风险前移表达**
3. **prepare → train 主链路**
4. **真实训练 lifecycle 管理**
5. **DeepSeek / Gemma 双 provider 下的复杂训练链路**
6. **高风险二次确认**

---

### 4.2 当前仍明显有边界的能力
以下问题这轮依然清晰存在：

1. **Gemma 的解释层不够 grounded**
   - 会把数字说虚
   - 会把类别名说错
   - 会把路径说成不存在的路径

2. **Gemma 的取消后参数回忆不够精确**
   - 大概知道是“什么任务”
   - 但具体参数会漂

3. **开放式总结任务比执行型任务更脆**
   - 做事时能靠工具约束
   - 总结时更容易离开工具事实自由发挥

---

## 5. 当前能力范围结论

### 你现在可以把它当什么来用
我会把它定义成：

> **一个已经能在真实脏数据场景下完成“数据准备 → 训练管理”主链路的工程化 Agent 原型。**

尤其是：
- 如果你关心的是“它能不能帮我干活”
- 而不是“它讲解时是不是每句话都特别严谨”

那么当前版本已经相当接近实用。

### 它现在最像什么
- **Gemma 路线：** 执行型 Agent，可用，但解释层需要继续收口
- **DeepSeek 路线：** 当前更像“可作为主 API provider 的稳定路线”

---

## 6. 下一步主线建议

基于这 10 方法测试，我建议主线下一步继续收这 3 件事：

### 1. 做“解释层 grounded 化”
重点对象：Gemma

思路：
- 对扫描/校验/prepare 结果增加更强的结果摘要模板
- 回答优先复用 tool 的 `summary / warnings / classes / resolved_args`
- 减少开放式自由发挥

### 2. 做“取消后参数回忆的结构化摘要”
当前 Memory 已经有 pending confirmation，但还不够“精确复读参数”。

应该补：
- 最近一次待确认操作摘要
- 最近一次准备完成后的 `data_yaml / model / epochs / device_source`

### 3. 继续保持“先主线稳，再扩功能”
现在已经到了**可以谨慎加功能**的门口，
但如果要让系统更像长期可用工具，我仍建议先把上面两点再收一轮。

---

## 7. 一句话结论

> **这轮 10 方法测试证明：当前系统在 `zyb` 这种“大数据量 + 大量缺失标签 + classes.txt 混入 labels 目录”的真实脏数据场景下，主链路已经能打；真正的边界不再是“会不会做事”，而是“Gemma 在解释层和参数回忆上还不够稳”。**
