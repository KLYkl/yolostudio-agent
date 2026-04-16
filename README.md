# YoloStudio Agent

[简体中文](./README.zh-CN.md)

YoloStudio Agent is the agent / MCP workspace behind YoloStudio's conversational training and prediction workflows.
It packages the MCP server, the dialogue client, rule-based knowledge, and the remote validation harnesses used to turn desktop-style YOLO operations into reproducible agent flows.

## Why this repo exists

This repository focuses on the **agent side** of YoloStudio:

- expose MCP tools for dataset, training, prediction, extraction, and knowledge lookup
- route natural-language requests into executable tool plans
- keep grounded state for follow-up questions, confirmations, summaries, and recommendations
- validate the same flows locally and through remote roundtrips

It is intentionally separate from the desktop product code so the agent / MCP layer can evolve, test, and publish independently.

## Capability snapshot

### Training workflows

- dataset readiness and training preflight
- dataset preparation / YAML regeneration
- training start / stop / status / summarize / inspect
- training outcome analysis and next-step recommendations
- training artifact return and remote follow-up validation

### Prediction workflows

- image and video prediction
- prediction summary and grounded follow-up replies
- prediction result management and export helpers
- realtime / RTSP prediction workflows

### Agent workflows

- intent parsing and tool routing
- confirmation / guardrail handling
- session state and grounded replies
- rule-based knowledge explanations for training decisions

## Repository layout

```text
agent/                      Source-of-truth agent client, server, tools, and tests
knowledge/                  Rule-based knowledge base used by the agent
deploy/scripts/             Remote validation, roundtrip, and bridge helpers
  deploy/server_proto/        Managed runtime mirror for remote deployment flows
docs/                       Selected workflow notes and test plans
```

## What is in scope today

The current public mainline centers on three paths:

1. **Training mainline**  
   readiness -> prepare/preflight -> start -> status -> summarize -> analyze -> recommend
2. **Prediction mainline**  
   image/video prediction -> summary -> result follow-up -> grounded replies
3. **Remote and realtime validation**  
   remote roundtrips, training artifact return, and RTSP / realtime prediction checks

## Quick start

### 1. Install dependencies

Client-side dependencies:

```bash
pip install -r agent/client/requirements_client.txt
```

Server-side dependencies:

```bash
pip install -r agent/server/requirements_server.txt
```

### 2. Start the MCP server

```bash
python -m yolostudio_agent.agent.server.mcp_server
```

On a remote Linux host, you can also use the helper script:

```bash
APP_ROOT=/opt/yolostudio-agent CONDA_BIN=/opt/conda/bin/conda ENV_NAME=agent-server bash deploy/scripts/manage_mcp_server.sh restart
```

### 3. Run a representative validation slice

```bash
python agent/tests/test_training_mainline_roundtrip.py
python agent/tests/test_prediction_route.py
python agent/tests/test_realtime_prediction_route.py
python agent/tests/test_remote_transfer_route.py
```

On a Windows client environment, a longer dialogue regression is commonly run with the local virtual environment:

```powershell
.\agent\.venv\Scripts\python.exe .\agent\tests\test_extreme_chat_regression.py
```

## Remote validation helpers

The scripts under `deploy/scripts/` are publication-safe examples. Their defaults are intentionally generic and should be overridden for your own environment.

Typical overrides include:

- SSH host alias or remote target
- remote app root
- remote dataset / model paths
- local staging / output paths
- conda environment name

Representative helpers:

- `deploy/scripts/run_training_remote_roundtrip.ps1`
- `deploy/scripts/run_realtime_rtsp_remote_roundtrip.ps1`
- `deploy/scripts/run_prediction_remote_roundtrip.ps1`
- `deploy/scripts/run_training_agent_remote_followup_matrix.sh`

## Path conventions

Public docs and scripts use generic example values such as:

- `C:\workspace\yolodo2.0`
- `/opt/yolostudio-agent`
- `/opt/conda`
- `/data/example_dataset`
- `/models/yolov8n.pt`
- `remote-agent`
- `203.0.113.10`

Replace them with values that match your own environment.

## Notes

- `agent/` is the source-of-truth implementation; the managed runtime mirror is refreshed with `python deploy/scripts/sync_server_proto.py`.
- `deploy/server_proto/` is not a hand-edited code source. Only deploy-only paths such as `.venv`, `core/`, `utils/`, and the mirror README stay local to that package.
- `docs/` only contains selected public notes / plans suitable for the repository.
- Local machine-specific paths, hosts, and secrets are intentionally sanitized before publication.
