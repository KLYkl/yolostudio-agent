from __future__ import annotations

from pathlib import Path

# Compatibility bridge for the legacy package name.
_ROOT = Path(__file__).resolve().parent.parent
__path__ = [str(_ROOT)]
