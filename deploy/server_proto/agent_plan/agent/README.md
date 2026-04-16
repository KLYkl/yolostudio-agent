# Agent Plan Workspace (Server Prototype)

本目录是远端部署原型中的 Agent 工作区镜像。

- 用于部署、回归和远端验证
- `client/`、`server/`、`tests/`、`AGENT.md` 等受控路径由 `python deploy/scripts/sync_server_proto.py` 从主真源刷新
- `core/`、`utils/`、`.venv` 等 deploy-only 路径不参与主真源镜像同步
- 示例路径与主仓库一样使用公开版占位默认值，请按实际环境覆盖
