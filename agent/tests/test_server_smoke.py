from __future__ import annotations

import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.services.train_service import TrainService
from agent_plan.agent.server.tools.train_tools import check_training_status, stop_training
from agent_plan.agent.server.tools.data_tools import scan_dataset


def main() -> None:
    service = TrainService()

    original_policy = os.environ.get("YOLOSTUDIO_TRAIN_DEVICE_POLICY")
    try:
        os.environ["YOLOSTUDIO_TRAIN_DEVICE_POLICY"] = "single_idle_gpu"
        print("resolve:auto(single_idle_gpu)", service._resolve_device("auto"))
        print("resolve:multi(single_idle_gpu)", service._resolve_device("0,1"))
        print("resolve:cpu", service._resolve_device("cpu"))

        os.environ["YOLOSTUDIO_TRAIN_DEVICE_POLICY"] = "all_idle_gpus"
        print("resolve:auto(all_idle_gpus)", service._resolve_device("auto"))
    finally:
        if original_policy is None:
            os.environ.pop("YOLOSTUDIO_TRAIN_DEVICE_POLICY", None)
        else:
            os.environ["YOLOSTUDIO_TRAIN_DEVICE_POLICY"] = original_policy

    missing_yaml = str(Path("C:/workspace/yolodo2.0/agent_plan/definitely_missing.yaml"))
    invalid = service.start(model="yolov8n.pt", data_yaml=missing_yaml, epochs=1)
    print("start:missing_yaml", invalid)

    print("tool:status", check_training_status())
    print("tool:stop", stop_training())
    print("tool:scan_missing", scan_dataset("Z:/definitely-not-exist"))


if __name__ == "__main__":
    main()
