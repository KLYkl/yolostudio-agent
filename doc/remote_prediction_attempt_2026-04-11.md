# 第二主线远端 prediction 实测尝试（2026-04-11）

> 注：本文件记录的是**当时失败的那一轮尝试**。
> 2026-04-11 后续已经完成远端真实验证，最新结果请看：`doc/prediction_remote_real_media_validation_2026-04-11.md`

## 目标

把第二主线继续推进到：

- 本地挑选权重与视频样本
- 传输到服务器
- 在服务器真实环境执行 prediction tool
- 得到远端真实 prediction 报告

## 本轮实际动作

1. 已有本地素材 staging：
   - `C:\workspace\yolodo2.0\agent_plan\deploy\scripts\stage_prediction_real_media.py`
   - staged 目录：`C:\workspace\yolodo2.0\agent_plan\.tmp_prediction_real_media_stage`
2. 已有远端上传脚本：
   - `C:\workspace\yolodo2.0\agent_plan\deploy\scripts\upload_prediction_real_media.ps1`
3. 已有远端执行脚本：
   - `C:\workspace\yolodo2.0\agent_plan\deploy\scripts\run_prediction_remote_validation.sh`
4. 已有远端测试脚本：
   - `C:\workspace\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media.py`
5. 新增远端连通性预检查：
   - `C:\workspace\yolodo2.0\agent_plan\deploy\scripts\check_remote_prediction_prereqs.ps1`

## 实测结果

执行：

```powershell
C:\workspace\yolodo2.0\agent_plan\deploy\scripts\check_remote_prediction_prereqs.ps1
```

得到：

- `203.0.113.10:22` -> `访问权限不允许`
- `203.0.113.10:8080` -> `访问权限不允许`
- `203.0.113.10:11434` -> `访问权限不允许`
- `ssh` 可执行文件存在，但 `ssh_exit=255`

## 结论

> 当前阻塞点已经进一步收敛：TCP 22 端口已通，但当前 Codex 进程既不能读取 `C:\Users\Public\.ssh\id_ed25519` 私钥文件，也不能连接到 Windows `ssh-agent`（`ssh-add -l` 返回 `Error connecting to agent: Permission denied`），因此仍无法完成 SSH/SCP 认证。

因此本轮**未能真正完成**：
- 上传 staged 权重 / 视频到服务器
- 在服务器运行真实 prediction

## 下一步恢复方式

只要换到一个允许对 `203.0.113.10` 发起 TCP / SSH 连接的终端，就按下面顺序继续：

1. 运行：
   - `C:\workspace\yolodo2.0\agent_plan\deploy\scripts\check_remote_prediction_prereqs.ps1`
2. 若通过，则上传：
   - `C:\workspace\yolodo2.0\agent_plan\deploy\scripts\upload_prediction_real_media.ps1 -Server remote-agent`
3. 再在远端执行：
   - `C:\workspace\yolodo2.0\agent_plan\deploy\scripts\run_prediction_remote_validation.sh`

## 当前策略

在远端 TCP 仍然被拦截之前，第二主线继续以：

- 本地 `yolo / yolodo` conda 环境真实验证

作为主验证路径。


补充定位：
- `ssh` 在禁用 host key 校验后，已能走到认证阶段
- 但会报：`Load key "C:\Users\Public\.ssh\id_ed25519": Permission denied`
- 同时 `ssh-add -l` 在当前进程里返回：`Error connecting to agent: Permission denied`
