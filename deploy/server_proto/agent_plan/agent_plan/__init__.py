from __future__ import annotations

import warnings
from pathlib import Path

# Compatibility bridge for the legacy package name.
_ROOT = Path(__file__).resolve().parent.parent
__path__ = [str(_ROOT)]

warnings.warn(
    "agent_plan is deprecated; use yolostudio_agent instead.",
    DeprecationWarning,
    stacklevel=2,
)
