# 远端传输与主线闭环能力说明（2026-04-13）

## 1. 新增能力
- `list_remote_profiles`
  - 列出当前可用的远端 profile 配置和可识别的 SSH alias。
- `upload_assets_to_remote`
  - 在 **本机** 把文件或目录上传到远端服务器。
  - 适用于权重、数据集、视频或其他运行输入。
  - 默认开启：
    - 大文件自动分块
    - 断点续传
    - 上传后哈希校验
    - 本机进度输出
- `download_assets_from_remote`
  - 在 **本机** 把远端服务器上的文件或目录下载回本地。
  - 适用于 prediction 输出、报告、清单或其他远端产物回传。

- 聊天主线新增两个统一闭环：
  - **远端预测闭环**
    - 上传本地模型 + 图片/视频
    - 远端执行 `predict_images` / `predict_videos`
    - 自动把结果下载回本机
  - **远端训练闭环**
    - 上传本地模型 + 数据集
    - 远端执行 `training_readiness`
    - 必要时自动 `prepare_dataset_for_training`
    - 再做 `training_preflight`
    - 最后 `start_training`
    - 如果用户明确要求“等训练结束后回传产物”，会继续：
      - 轮询 `check_training_status`
      - 汇总 `summarize_training_run`
      - 解析远端 `save_dir`
      - 自动把 run 目录下载回本机

> 这不是服务器端自己读取本机路径；真正的数据传输发生在客户端本机。

## 2. 默认配置搜索路径
按优先级：
1. 显式传入 `profiles_path`
2. 环境变量 `YOLOSTUDIO_REMOTE_PROFILES_PATH`
3. `D:\yolodo2.0\agent_plan\.codex\remote_profiles.json`
4. `~/.yolostudio/remote_profiles.json`

示例格式见：
- `D:\yolodo2.0\agent_plan\deploy\examples\remote_profiles.example.json`

## 3. 聊天层行为
- 当用户问“有哪些可用服务器 / 远端配置 / SSH alias”时，Agent 会优先走 `list_remote_profiles`。
- 当用户要求“把本机权重 / 数据 / 视频上传到服务器”时，Agent 会组装：
  - 本地路径
  - 目标服务器 / profile
  - 远端目录
- 因为这是外部传输，聊天层会先进入确认，再执行 `upload_assets_to_remote`。
- 当用户要求“上传后直接去远端跑 prediction 并把结果拉回本机”时，Agent 会进入 **远端预测闭环**。
- 当用户要求“上传后直接去远端开训”时，Agent 会进入 **远端训练闭环**。
- 当用户明确说“等训练结束 / 训练完后拉回结果 / 回传训练产物”时，Agent 会把远端训练闭环切到 **等待完成并自动回传** 模式。

确认文案会额外提示：
- 当前阈值（默认 `256MB`）
- 分块大小（默认 `64MB`）
- 默认会开启断点续传和哈希校验
- 远端预测闭环当前要求待预测输入是**单个文件或单个目录**；多个散文件应先整理进目录。
- 远端训练闭环默认先快速启动训练，不会无条件长时间阻塞；只有用户明确要求等待结束并回传时，才进入轮询等待模式。

## 4. 本次验证
### 4.1 本地单测
通过：
- `D:\yolodo2.0\agent_plan\agent\tests\test_remote_transfer_tools.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_remote_transfer_route.py`
- `D:\yolodo2.0\agent_plan\agent\tests\test_tool_alias_adapter.py`

### 4.2 实机 smoke
使用本机 `.venv` Python 调用 `upload_assets_to_remote`，目标使用 SSH alias `yolostudio`，把临时文件上传到远端 `/tmp/codex_remote_transfer_smoke`；随后远端 `ls/cat` 验证文件已落地并可读取。

本次增强后补充验证目标：
- 低阈值强制进入 chunked 模式
- 断点续传复用已存在 chunk
- 上传完成后做 `sha256` 校验
- 第二次重复上传时直接复用远端同 hash 文件

本次补充实测：
- 本机构造 `3MB` 临时文件，设置：
  - `large_file_threshold_mb=1`
  - `chunk_size_mb=1`
- 第一次上传：
  - `chunked_file_count = 1`
  - `verified_file_count = 1`
  - `transferred_bytes = 3145728`
- 第二次上传同一文件到同一路径：
  - `skipped_file_count = 1`
  - `transferred_bytes = 0`
  - `skipped_bytes = 3145728`

本次新增补充实测：
- 使用本机 `.venv` Python 调用：
  - `upload_assets_to_remote(local_paths=[payload.txt], server='yolostudio', remote_root='/tmp/codex_remote_pipeline_smoke')`
  - `download_assets_from_remote(remote_paths=['/tmp/codex_remote_pipeline_smoke/payload.txt'], server='yolostudio')`
- 结果：
- 上传成功
- 下载成功
- 回传文件内容与本地原始内容一致（`MATCH=True`）

本次新增主线回归：
- 本地通过：
  - `D:\yolodo2.0\agent_plan\agent\tests\test_remote_transfer_tools.py`
  - `D:\yolodo2.0\agent_plan\agent\tests\test_remote_transfer_route.py`
  - `D:\yolodo2.0\agent_plan\agent\tests\test_tool_alias_adapter.py`
  - `D:\yolodo2.0\agent_plan\agent\tests\test_training_rules_contract.py`
  - `D:\yolodo2.0\agent_plan\agent\tests\test_predict_tools.py`
  - `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_management_route.py`
- 远端 `yolostudio` 通过：
  - `test_training_run_summary.py`
  - `test_training_run_inspect.py`
  - `test_training_run_list.py`
  - `test_tool_alias_adapter.py`

补充说明：
- `test_training_run_summary.py` 已改成使用 `__file__` 相对临时目录，不再硬编码 `/mnt/d/...`，可在 Linux 节点直接跑。
- 训练 run 相关测试在本机 Windows 环境里仍可能被进程终止权限问题干扰，所以这轮把环境敏感回归放到了 `yolostudio` 节点执行。
- `extract_remote_root_from_text` 已修正：当一句话里同时出现 Linux 本地绝对路径和远端绝对路径时，会优先识别真正的远端目录，不再把本地 `/home/...` 或 `/tmp/...` 路径误当成 `remote_root`。

过程中还修复了一个真实 smoke 暴露的问题：
- 远端组包脚本原先会把 `part_*.sha256` 也误拼进最终文件
- 现已改成只拼接 `part_数字` 正文分块

结论：
- 新能力已不再依赖当前机器硬编码脚本。
- 只要目标机器具备 `ssh/scp`，并提供 SSH alias 或 profile 配置，就能在本机侧完成上传/下载。
- 主线现在已经能把“上传 -> 远端预测 -> 下载结果”以及“上传 -> 远端训练启动”收成单次聊天闭环。
- 主线现在已经能把“上传 -> 远端训练 -> 等待完成 -> 自动回传 run 目录”也收成单次聊天闭环。
- 当前版本已经适合几百 MB 到若干 GB 的常规工程传输；链路中断时可复用已存在分块，但还没做 rsync 级别的双端扫描优化。
