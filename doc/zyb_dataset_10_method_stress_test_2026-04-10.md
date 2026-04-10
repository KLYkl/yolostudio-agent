# zyb 大数据集 10 方法高强度测试（2026-04-10）

## 1. 测试目标

本轮测试不再只验证“小而标准的数据集”，而是用一份更大、也更脏的真实数据来判断当前 Agent 的能力边界。

测试目标：

1. 验证当前系统在 **大数据量** 下的扫描、校验、准备、训练链路是否还能工作
2. 验证当前 Agent 在 **复杂自然语言意图** 下是否还能稳定落到正确工具链
3. 找出当前版本最容易暴露的问题，不假设“一定没问题”

---

## 2. 测试对象概览

本轮使用的数据集代号：`zyb`

结构：
- dataset root
  - `images/`
  - `labels/`

快速统计：
- 图片数：`7027`
- 标签文件数：`1849`
- 同名匹配图片：`1848`
- 缺失标签图片：`5179`
- 空标签文件：`1`
- 孤儿标签：`1`
- labels 目录额外存在：`classes.txt`

补充说明：
- 当前系统能把它识别为标准 YOLO 根目录
- 但它不是“干净训练集”，更像一份适合压测鲁棒性的真实数据

---

## 3. 10 个测试方法

| # | 方法 | 类型 | 目标 |
|---|---|---|---|
| 1 | root 路径扫描 | Tool | 验证大数据量下 root 解析与统计是否正确 |
| 2 | 标签合法性校验 | Tool | 验证 dirty dataset 下 validate 的边界 |
| 3 | 训练就绪判断 | Tool | 验证 readiness 是否给出合理阻塞原因 |
| 4 | 默认准备流程 | Tool | 验证 prepare 是否能把 raw dataset 推到可训练状态 |
| 5 | 准备后训练冒烟 | Tool | 验证这份大数据是否真的能启动训练 |
| 6 | Agent 扫描总结 | Agent / Gemma | 验证 Agent 对大数据扫描结果的总结能力 |
| 7 | Agent 只判断、不训练 | Agent / Gemma | 验证“只检查，不执行”约束 |
| 8 | Agent 脏数据诊断 | Agent / Gemma | 验证对 `classes.txt` / 脏目录细节的理解能力 |
| 9 | Agent 只准备不训练 | Agent / Gemma | 验证高风险 prepare-only 场景 |
| 10 | Agent 复杂训练链（含 provider 对照） | Agent / Gemma + DeepSeek | 验证 root → prepare → train 的复杂链路与 provider 差异 |

---

## 4. 逐项结果

### 方法 1：root 路径扫描
**输入**：直接把 dataset root 交给 `scan_dataset`

**结果**：通过
- 能正确解析到 `images/` 与 `labels/`
- 返回：
  - `总图片: 7027`
  - `已标注: 1848`
  - `缺失标签: 5179`
  - `空标签: 1`
  - `类别数: 4`

**判断**：
- 当前 root resolver 在大数据量下仍稳定
- 这一点说明主线的 root 语义修复是有效的

---

### 方法 2：标签合法性校验
**输入**：直接对 dataset root 调 `validate_dataset`

**结果**：表面通过，但暴露语义边界
- 返回：`未发现标签问题`
- `issue_count = 0`

**暴露的问题**：
- 这并不代表数据“干净”
- 它只说明：**已有标签文件本身格式合法**
- 但这份数据同时仍有：
  - `5179` 张图片无同名标签
  - `labels/classes.txt` 混在标签目录里

**判断**：
- 当前 `validate_dataset` 更像“标签文件格式校验”
- 不是“数据集完整性校验”
- 这是一个明确的能力边界

---

### 方法 3：训练就绪判断
**输入**：`training_readiness(dataset_root)`

**结果**：通过，但偏乐观
- 结论：`ready = false`
- 阻塞原因：`缺少可用的 data_yaml`
- 没有因为 5179 张未标注图片而阻塞

**判断**：
- readiness 当前会把“缺 yaml”当主阻塞项
- 但不会把“大量未标注图片”视为失败
- 如果这些未标注图片其实是背景图，这可能没问题；如果是漏标，则 readiness 过于乐观

**结论**：
- 这是当前系统一个很重要的灰区：
  > **readiness 对“缺失标签很多”没有显式风险提示**

---

### 方法 4：默认准备流程
**输入**：`prepare_dataset_for_training(dataset_root)`

**结果**：通过
- 自动完成：
  1. resolve root
  2. scan
  3. validate
  4. split
  5. generate_yaml
  6. readiness
- 最终输出：
  - `ready = true`
  - 生成 YAML：`images_split/data.yaml`

**额外观察**：
- 本轮 prepare 因为缺 yaml，自动触发了 split
- split 后统计：
  - `train = 5621`
  - `val = 1406`
- 也就是说它按 7027 张图片整体进行了划分

**判断**：
- 当前 prepare 主线在大数据集上是能跑通的
- 但它对“缺失标签很多”的数据集也会直接推进，这偏激进

---

### 方法 5：准备后训练冒烟
**输入**：
- `model = yolov8n`
- `data = images_split/data.yaml`
- `epochs = 2`

**结果**：通过
- 训练成功启动
- 6 秒后状态查询：`running = true`
- 随后 stop 成功：`return_code = -15`

**判断**：
- 这说明当前系统不仅能“准备数据”，还能真的把这份大数据带进训练链路
- 从主线角度，这是一条非常有价值的验证：
  > **当前主线对大数据 dirty dataset 不是停留在纸面，而是真的能拉起训练**

---

### 方法 6：Agent 扫描总结（Gemma）
**提示词**：
> 请扫描该数据集，并总结图片数、标签数、缺失标签、类别情况。

**结果**：通过
- 能正确总结：
  - 7027 张图
  - 1848 已标注
  - 5179 缺失标签
  - 4 类
- 能明确说明当前路径是 root，并解析到了 `images/labels`

**判断**：
- 对“扫描并解释结果”这类任务，Gemma 已经能胜任
- 属于当前版本的强项

---

### 方法 7：Agent 只判断、不训练（Gemma）
**提示词**：
> 请判断该数据集能不能直接训练，不要启动训练，只告诉我原因和下一步建议。

**结果**：通过
- 它没有误触发训练
- 给出的主结论是：当前缺少可用的 `data.yaml`
- 推荐先生成 YAML

**判断**：
- “只检查不执行”这个约束是稳的
- 但解释仍偏单一：只强调 yaml，不强调大量未标注图片的风险

---

### 方法 8：Agent 脏数据诊断（Gemma）
**提示词**：
> labels 目录里可能混有 classes.txt。请检查这个数据集，判断这会不会影响训练准备，并给出处理建议，但不要启动训练。

**结果**：部分失败

**表现**：
- 它没有把 `classes.txt` 识别成当前需要重点解释的风险点
- 反而给出了一个并不存在于当前系统中的建议：`generate_data.py`
- 还倾向于让用户“忽略 classes.txt 顾虑”

**判断**：
- 这是本轮最有代表性的 Agent 理解短板之一：
  > **当用户问的是“脏数据细节怎么处理”时，Gemma 仍会产生脱离当前系统实际能力的建议。**

**这不是工具层失败，而是 Agent 解释层失败。**

---

### 方法 9：Agent 只准备不训练（Gemma）
**提示词**：
> 数据在该目录，按默认划分比例准备成可训练状态，但不要开始训练。

**结果**：部分通过
- 第一阶段行为是正确的：
  - 命中 `prepare_dataset_for_training`
  - 走了高风险确认
- 工具结果本身也对：
  - 成功得到可训练 YAML

**但自然语言回复有问题**：
- 回复中出现了“训练集/验证集/测试集”“评估模型”“查看报告”等扩展叙述
- 这些内容并不是这次工具真实做过的事情

**判断**：
- prepare-only 主链路本身是通的
- 但 Gemma 在**完成后的自然语言包装**上仍会说得过头

这是当前非常明确的一条边界：
> **工具执行正确 ≠ 模型解释绝对可信**

---

### 方法 10：Agent 复杂训练链（Gemma + DeepSeek 对照）
**提示词**：
> 数据在该目录，按默认划分比例，然后用 yolov8n 模型训练 2 轮。

#### Gemma 结果
通过
- 第一次确认：`prepare_dataset_for_training(force_split=true)`
- 第二次确认：`start_training(data_yaml=..., model=yolov8n.pt, epochs=2)`
- 确认后训练成功启动
- 随后我又查询状态并 stop，均成功

#### DeepSeek 结果
通过
- 第一次确认：`prepare_dataset_for_training(force_split=true, split_ratio=0.8)`
- 第二次确认：`start_training(model=yolov8n, data_yaml=..., epochs=2, device=auto)`
- 我在第二次确认时取消，用于保留 provider 行为证据

**对照判断**：
- 两个 provider 都已经能把复杂自然语言落回：
  - `prepare -> train`
  两段式高风险流程
- DeepSeek 的参数保留更“原味”
- Gemma 更依赖当前主线 fallback 才稳定

---

## 5. 本轮新增暴露的问题

### 问题 A：`validate_dataset` 与 `training_readiness` 对“大量缺失标签”的风险表达不足
当前现象：
- `scan_dataset` 能清楚看见 `5179` 张图片缺失标签
- 但 `validate_dataset` 仍返回“未发现标签问题”
- `training_readiness` 也不会把这点作为 blocker

这意味着：
> **系统目前能看见“缺失标签很多”，但还不会把它稳定地升级成“训练风险提示”。**

---

### 问题 B：自动生成的 YAML 使用了数字类名，而不是 `classes.txt` 中的真实类名
我实际读取了 prepare 生成的 YAML，内容是：

```yaml
names:
  0: '0'
  1: '1'
  2: '2'
  3: '3'
```

而不是：
- Excavator
- bulldozer
- piling_machine
- two_wheeler

这说明：
> **prepare 流程目前能生成可训练 YAML，但类名语义会丢失。**

这对“能不能训练”影响不一定致命，
但对“训练结果可解释性”和“后续展示”会有明显影响。

---

### 问题 C：Gemma 在“脏数据解释”与“prepare 后说明”上仍会说过头
本轮两个典型表现：
1. 把不存在的 `generate_data.py` 说成建议
2. 在 prepare-only 场景下扩展到“评估”“报告”“测试集”等未实际发生的内容

这说明：
> **当前 Gemma 的执行链路已经比较稳，但解释层还不够收。**

---

## 6. 能力范围判断

### 当前已经相当稳的范围
1. 标准 root 目录识别
2. 大数据量扫描与摘要
3. 训练前检查
4. prepare → train 两段式高风险流程
5. 大数据集上的真实训练启动、状态查询、停止
6. Gemma / DeepSeek 双 provider 下的主链路训练意图

### 当前仍有明显边界的范围
1. 脏数据风险表达（特别是大量未标注图片）
2. 利用 `classes.txt` 保留语义类名
3. Gemma 对执行后结果的解释收口
4. “工具真实结果”和“模型自然语言解释”之间的精确对齐

---

## 7. 结论

这份 `zyb` 数据集非常值得继续保留作为主线回归集。

因为它不是“完美数据”，而是正好能暴露当前系统最真实的边界：
- 大数据量
- 缺失标签很多
- labels 中混有 `classes.txt`
- prepare 能跑通，但语义细节会丢失
- Agent 主链路已稳，但解释层还不够稳

如果只给一句总结：

> **当前版本已经能把这份大数据脏数据集真正带进 prepare 和 training 主线，但它对“数据质量风险”和“类名语义保留”的处理还不够成熟，Gemma 的解释层也还需要继续收口。**

---

## 8. 产物

- 原始结构化结果：`agent/tests/test_zyb_large_dataset_output.json`
- 测试脚本：`agent/tests/test_zyb_large_dataset_e2e.py`
