# YoloStudio Agent

[简体中文](./README.zh-CN.md)

YoloStudio Agent is the agent / MCP runtime behind YoloStudio's conversational YOLO workflows.
It packages the dialogue client, the FastMCP tool server, the rule-based knowledge layer, and the remote validation harnesses that turn dataset preparation, training, prediction, and result follow-up into reproducible agent flows.

**Project reference:** [KLYkl/YoloStudio](https://github.com/KLYkl/YoloStudio)

This repository is an independent agent / MCP workspace developed based on YoloStudio.

The referenced YoloStudio repository is the product-facing desktop application repository: a PySide6 desktop app for YOLO dataset preparation, model training, and visual inference workflows.
This repository reuses that product context as its origin and domain background, but it is maintained as a separate codebase with its own agent-runtime focus.

This repository is **not** the desktop UI itself.
It is the agent-side workspace used to:

- expose YOLO-related capabilities as MCP tools
- route natural-language requests into executable tool flows
- keep grounded session state for follow-up dialogue
- enforce confirmation / HITL boundaries for risky actions
- validate the same workflows locally and on remote GPU hosts

## Relationship to YoloStudio

If you are arriving from the YoloStudio repository, the relationship is better understood as:

- **Reference / origin repo**: [KLYkl/YoloStudio](https://github.com/KLYkl/YoloStudio)  
  Focuses on the PySide6 desktop product and its user-facing application modules.
- **This repo (`agent_plan`)**: an independent agent / MCP runtime workspace  
  Focuses on dialogue orchestration, MCP tool exposure, knowledge-backed follow-up, and remote validation flows around the same YOLO-oriented problem domain.

In short: YoloStudio is the upstream product/background reference, while this repository is a separate agent-focused codebase built on top of that context rather than a submodule or same-repo execution layer.

## Why this repo exists

The agent layer evolves at a different pace from the desktop product.
Keeping it separate makes it easier to:

- iterate on tool schemas, routing, memory, and confirmation behavior
- validate LLM-first workflows without coupling every change to the desktop release cycle
- run remote roundtrip and regression suites against controlled server mirrors
- publish a sanitized, reviewable agent workspace without shipping product-internal paths or credentials

## What the agent can do today

### 1. Dataset and data-prep workflows

The MCP server exposes tools for common YOLO dataset operations, including:

- dataset scanning, validation, health checks, and duplicate detection
- train/val splitting and YAML generation
- label modification, orphan cleanup, empty-label generation, and missing-label generation
- format conversion and class-based categorization
- image extraction and video frame extraction
- combined `prepare_dataset_for_training` flow for common preflight preparation

### 2. Training workflows

The runtime supports a full training-oriented dialogue loop:

- dataset-readiness vs execution-readiness separation
- training environment discovery and GPU checks
- preflight before execution
- training start / status / stop
- run summarization, inspection, comparison, and best-run selection
- post-training analysis and next-step recommendations from the knowledge layer

### 3. Prediction workflows

Prediction is not limited to one-shot image inference. The current tool surface covers:

- image and video prediction
- asynchronous image prediction sessions
- prediction summary, output inspection, and report export
- prediction result organization and path-list export
- camera / screen / RTSP discovery and realtime prediction sessions
- realtime status polling and stop controls

### 4. Agent-runtime workflows

On the dialogue side, the client handles:

- intent parsing and mainline routing
- grounded follow-up replies from structured tool results
- session state retention across training / prediction / remote-transfer threads
- manual or automatic confirmation flow for destructive or long-running actions
- memory persistence for recent turns, events, and pending actions
- helper-LLM and loop-planner-LLM roles when explicitly enabled

### 5. Remote validation and deployment workflows

The repo also carries the operational tooling used to validate real roundtrips:

- sync the managed remote mirror under `deploy/server_proto/`
- start / stop / inspect the MCP server on a remote Linux host
- run remote training, prediction, realtime RTSP, and chaos-validation scripts
- keep example remote profiles and bridge helpers in version control

## Architecture at a glance

```text
User / CLI
  -> agent/client/
     - LangGraph-based runtime
     - session state, context retention, HITL, reply rendering
     - request services for training / prediction / follow-up flows
  -> MCP connection
  -> agent/server/mcp_server.py
     - FastMCP tool registration
     - read vs destructive tool annotations
     - optional loop planner wiring
  -> agent/server/tools/
     - data / extract / knowledge / predict / train / training_loop tools
  -> agent/server/services/
     - actual dataset, training, prediction, realtime, and GPU logic
  -> knowledge/
     - rule-based training guidance and recommendations
  -> deploy/
     - mirror sync, remote roundtrip scripts, managed server runtime
```

## Repository layout

```text
agent/
  client/                 Dialogue runtime, memory, routing, HITL, CLI
  server/                 FastMCP server, tool surface, backend services
  tests/                  Roundtrip, regression, chaos, and contract tests
  AGENT.md                Runtime-agent behavior contract
knowledge/                Rule-based knowledge base and playbooks
deploy/
  examples/               Example remote profile configs
  scripts/                Mirror sync, remote roundtrip, MCP management helpers
  server_proto/           Managed remote mirror + deploy-only runtime deps
    agent_plan/           Mirrored package used on remote hosts
    core/                 Deploy-only data/runtime helpers
    utils/                Deploy-only utility modules required by server runtime
docs/                     Current design notes and refactor reports
memory/                   Local session/event persistence during development
```

## Design principles reflected in the codebase

This repository is currently built around a few stable constraints:

- **LLM-first, tool-grounded**: the model decides with tool evidence, rather than hard-coded template routing everywhere.
- **Readiness separation**: dataset structure checks and actual training execution readiness are intentionally different steps.
- **Confirmation gates**: data-destructive actions, uploads, and long-running tasks are meant to go through pending-action review.
- **Grounded follow-up**: the client prefers reusing recent structured facts for “continue”, “what about just now”, and result-explanation turns.
- **Remote reproducibility**: remote validation should use the managed mirror, not an ad-hoc manually edited server copy.

If you want the reasoning behind the current direction, start with:

- `docs/agent-runtime-contract.md`
- `docs/agent-client-refactor-report.md`
- `docs/README.md`

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

### 2. Export the runtime import path

The current local source-tree startup needs both the repo root and `deploy/server_proto` on `PYTHONPATH`, because the server runtime still reuses deploy-only `core/` and `utils/` packages from the managed mirror.

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\deploy\server_proto"
```

bash:

```bash
export PYTHONPATH="$PWD:$PWD/deploy/server_proto"
```

### 3. Configure an LLM provider

The default path is local Ollama. Typical local setup:

PowerShell:

```powershell
$env:YOLOSTUDIO_LLM_PROVIDER = "ollama"
$env:YOLOSTUDIO_AGENT_MODEL = "gemma4:e4b"
$env:YOLOSTUDIO_OLLAMA_URL = "http://127.0.0.1:11434"
```

bash:

```bash
export YOLOSTUDIO_LLM_PROVIDER=ollama
export YOLOSTUDIO_AGENT_MODEL=gemma4:e4b
export YOLOSTUDIO_OLLAMA_URL=http://127.0.0.1:11434
```

The client also supports `deepseek` and `openai_compatible` providers via environment variables.

### 4. Start the MCP server

```bash
python -m yolostudio_agent.agent.server.mcp_server
```

The default MCP endpoint is `http://127.0.0.1:8080/mcp`.

For a remote Linux host, the managed helper is:

```bash
APP_ROOT=/opt/yolostudio-agent CONDA_BIN=/opt/conda/bin/conda ENV_NAME=agent-server bash deploy/scripts/manage_mcp_server.sh restart
```

### 5. Start the interactive agent CLI

```bash
python -m yolostudio_agent.agent.client.cli demo-session --confirm manual
```

Useful flags:

- `--confirm manual|auto`
- `--provider ollama|deepseek|openai_compatible`
- `--model <model-name>`
- `--base-url <llm-base-url>`

### 6. Try representative prompts

Examples that match the current repo focus:

- `Check whether this dataset is ready for training: <dataset_root>`
- `Prepare this dataset for YOLO training and regenerate data.yaml if needed`
- `Start training with yolov8n.pt on <data_yaml>`
- `Compare the latest two runs and tell me which one is better`
- `Run prediction on this image directory and summarize the results`
- `Test this RTSP stream and start realtime prediction if it is reachable`
- `Explain why the latest training run underperformed and suggest the next step`

## Important configuration knobs

These are the environment variables most people need first:

| Variable | Purpose |
| --- | --- |
| `YOLOSTUDIO_MCP_URL` | MCP endpoint used by the client. Default: `http://127.0.0.1:8080/mcp` |
| `YOLOSTUDIO_CONFIRMATION_MODE` | `manual` or `auto`; controls pending-action review behavior |
| `YOLOSTUDIO_LLM_PROVIDER` | Primary provider selection (`ollama`, `deepseek`, `openai_compatible`) |
| `YOLOSTUDIO_AGENT_MODEL` | Primary chat model |
| `YOLOSTUDIO_LLM_BASE_URL` / `YOLOSTUDIO_LLM_API_KEY` | Base URL and credentials for compatible providers |
| `YOLOSTUDIO_MEMORY_ROOT` | Where sessions/events/checkpoints are stored |
| `YOLOSTUDIO_MAX_HISTORY_MESSAGES` | Conversation-history retention window |
| `YOLOSTUDIO_HELPER_LLM_*` | Optional helper-model override for side tasks |
| `YOLOSTUDIO_LOOP_LLM_*` | Optional loop-planner model configuration |
| `YOLOSTUDIO_ENABLE_LOOP_PLANNER` | Enable host-side loop planner wiring |
| `YOLOSTUDIO_TRAIN_DEVICE_POLICY` | GPU allocation policy for training |
| `YOLOSTUDIO_GPU_BUSY_UTILIZATION_THRESHOLD_PCT` | Busy-GPU threshold tuning |
| `YOLOSTUDIO_GPU_BUSY_COMPUTE_MEMORY_THRESHOLD_MB` | Busy-GPU memory threshold tuning |
| `YOLOSTUDIO_REMOTE_PROFILES_PATH` | Path to remote profile configuration JSON |

## Validation and regression entry points

This repo is heavier on workflow validation than on marketing demos. Representative validation slices include:

### Metadata and server-surface checks

```bash
python agent/tests/test_mcp_server_metadata.py
python agent/tests/test_server_smoke.py
```

### Training and prediction mainlines

```bash
python agent/tests/test_training_mainline_roundtrip.py
python agent/tests/test_prediction_route.py
python agent/tests/test_realtime_prediction_route.py
python agent/tests/test_remote_transfer_route.py
```

### Longer dialogue regressions

```bash
python agent/tests/test_extreme_chat_regression.py
python agent/tests/test_training_loop_dialogue_matrix.py
```

### Chaos / resilience coverage

```bash
python agent/tests/test_agent_server_chaos_p0.py
python agent/tests/test_agent_server_chaos_p1_followup.py
python agent/tests/test_agent_server_chaos_p2_context.py
```

Some validation scripts assume prepared datasets, model weights, a reachable MCP server, or remote infrastructure. Read the test file before treating it as a one-command smoke check.

## Remote deployment and roundtrip workflow

The deployment story in this repo is based on a managed mirror, not direct rsync of the live dev tree.

Typical workflow:

1. Refresh the managed mirror:

   ```bash
   python deploy/scripts/sync_server_proto.py
   ```

2. Review the remote profile template:

   - `deploy/examples/remote_profiles.example.json`

3. Sync the mirror to the target machine and run a remote roundtrip script, for example:

   - `deploy/scripts/run_training_remote_roundtrip.ps1`
   - `deploy/scripts/run_prediction_remote_roundtrip.ps1`
   - `deploy/scripts/run_realtime_rtsp_remote_roundtrip.ps1`
   - `deploy/scripts/run_training_agent_remote_followup_matrix.sh`

4. Use `deploy/scripts/manage_mcp_server.sh` on the remote host to restart or inspect the server.

## Scope and non-goals

This repository intentionally focuses on the agent side of YoloStudio.

In scope:

- MCP tool exposure
- agent runtime behavior
- training / prediction / knowledge dialogue workflows
- remote validation and roundtrip reproducibility

Out of scope:

- the desktop GUI itself
- generic product marketing pages
- automatic promotion of test data into production knowledge
- shipping local machine paths, tokens, or unsanitized customer infrastructure details

## Notes on knowledge and data hygiene

The knowledge layer under `knowledge/` is intentionally tiered:

- `official` and `workflow` knowledge can participate in default guidance
- `case` knowledge is not auto-enabled by default
- `test` knowledge is blocked from production-style recommendation flows

This is important: regression output should not silently turn into “learned best practice”.

## Current docs worth reading

- `docs/agent-runtime-contract.md` — current runtime interface and pending-action contract
- `docs/agent-client-refactor-report.md` — current refactor rationale and direction; useful for design context, not as a line-accurate code map
- `docs/README.md` — which docs are current, historical, or archived
- `knowledge/README.md` — current knowledge-source policy and anti-pollution boundaries

## Publication / sanitization note

All public docs and scripts should keep paths, hosts, API keys, and environment-specific details sanitized.
Example values in this repository are placeholders and should be replaced for your own environment.
