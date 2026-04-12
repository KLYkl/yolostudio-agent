from __future__ import annotations

from pathlib import Path

# Bridge the public package name to the existing repository layout.
_ROOT = Path(__file__).resolve().parent.parent
__path__ = [str(_ROOT)]
