# YoloStudio Agent

[中文说明](./README.zh-CN.md)

YoloStudio Agent is the agent/MCP workspace for turning the desktop YoloStudio workflow into a conversational, remotely executable training and prediction system.

## What is included

- **MCP tool layer** for dataset inspection, data preparation, training, prediction, extraction, and training knowledge explanations
- **Agent client** for intent parsing, tool routing, confirmation handling, grounded replies, and session state
- **Remote validation scripts** for training and prediction roundtrips
- **Regression tests** covering training planning, training mainline roundtrips, prediction flows, and selected remote harnesses

## Repository layout

```text
agent/                  Agent client, server, tools, and tests
knowledge/              Phase 1 rule-based knowledge base
deploy/                 Remote validation scripts and server prototype
doc/                    Engineering notes, playbooks, and release context
```

## Current scope

This repository focuses on:

1. **Training mainline**: readiness -> prepare/preflight -> start -> status -> summarize -> analyze -> recommend
2. **Prediction mainline**: image/video prediction -> summary -> grounded replies
3. **Remote roundtrips**: validating the same workflows through a remote MCP server

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
python -m agent_plan.agent.server.mcp_server
```

Or use the helper script on a remote Linux host:

```bash
APP_ROOT=/opt/yolostudio-agent CONDA_BIN=/opt/conda/bin/conda ENV_NAME=agent-server bash deploy/scripts/manage_mcp_server.sh restart
```

### 3. Run a representative test slice

```bash
python agent/tests/test_training_mainline_roundtrip.py
python agent/tests/test_training_mainline_final_state_roundtrip.py
python agent/tests/test_training_status_route_phrases.py
```

On the Windows client environment, an additional long-dialogue regression is typically run with the client virtual environment:

```powershell
.\agent\.venv\Scripts\python.exe .\agent\tests\test_extreme_chat_regression.py
```

## Remote validation helpers

The scripts under `deploy/scripts/` are GitHub-safe examples. Their defaults are generic and should be overridden for your environment.

Typical overrides include:

- remote SSH target or host alias
- remote app root
- remote dataset/model locations
- local stage/output directories
- conda environment name

## Path conventions in docs and scripts

Public docs and scripts use generic example values such as:

- `C:\workspace\yolodo2.0`
- `/opt/yolostudio-agent`
- `/opt/conda`
- `/data/example_dataset`
- `/models/yolov8n.pt`
- `remote-agent`
- `203.0.113.10`

Replace them with paths and hosts that match your own environment.

## Notes

- This repo intentionally separates the **agent/MCP workspace** from the main desktop product code.
- Internal engineering notes remain under `doc/`, but sensitive local paths and host details have been sanitized for publication.
