# 第二主线远端 prediction 实测尝试（2026-04-11）

## 目标

把第二主线继续推进到：

- 本地挑选权重与视频样本
- 传输到服务器
- 在服务器真实环境执行 prediction tool
- 得到远端真实 prediction 报告

## 本轮实际动作

1. 已有本地素材 staging：
   - `D:\yolodo2.0\agent_plan\deploy\scripts\stage_prediction_real_media.py`
   - staged 目录：`D:\yolodo2.0\agent_plan\.tmp_prediction_real_media_stage`
2. 已有远端上传脚本：
   - `D:\yolodo2.0\agent_plan\deploy\scripts\upload_prediction_real_media.ps1`
3. 已有远端执行脚本：
   - `D:\yolodo2.0\agent_plan\deploy\scripts\run_prediction_remote_validation.sh`
4. 已有远端测试脚本：
   - `D:\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media.py`
5. 新增远端连通性预检查：
   - `D:\yolodo2.0\agent_plan\deploy\scripts\check_remote_prediction_prereqs.ps1`

## 实测结果

执行：

```powershell
D:\yolodo2.0\agent_plan\deploy\scripts\check_remote_prediction_prereqs.ps1
```

得到：

- `192.168.0.163:22` -> `访问权限不允许`
- `192.168.0.163:8080` -> `访问权限不允许`
- `192.168.0.163:11434` -> `访问权限不允许`
- `ssh` 可执行文件存在，但 `ssh_exit=255`

## 结论

> 当前阻塞点不在 `agent_plan` 代码，也不在远端验证脚本，而在当前 Codex / PowerShell 运行环境的 TCP 出站能力。

因此本轮**未能真正完成**：
- 上传 staged 权重 / 视频到服务器
- 在服务器运行真实 prediction

## 下一步恢复方式

只要换到一个允许对 `192.168.0.163` 发起 TCP / SSH 连接的终端，就按下面顺序继续：

1. 运行：
   - `D:\yolodo2.0\agent_plan\deploy\scripts\check_remote_prediction_prereqs.ps1`
2. 若通过，则上传：
   - `D:\yolodo2.0\agent_plan\deploy\scripts\upload_prediction_real_media.ps1 -Server yolostudio`
3. 再在远端执行：
   - `D:\yolodo2.0\agent_plan\deploy\scripts\run_prediction_remote_validation.sh`

## 当前策略

在远端 TCP 仍然被拦截之前，第二主线继续以：

- 本地 `yolo / yolodo` conda 环境真实验证

作为主验证路径。
