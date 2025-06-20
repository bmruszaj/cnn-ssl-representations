# utils/params_yaml.py
"""Robust helper to load *params.yaml* from project root.

Assumes this file lives in *src/utils/*. It climbs three levels up to
reach the repository root (../..). If you move this file, adjust the
`_project_root()` helper accordingly or fall back to an upward search.
"""

import yaml
from typing import Any, Dict
from pathlib import Path


def _project_root() -> Path:
    """Return path to the repository root relative to this file."""
    # src/utils/params_yaml.py -> src/utils -> src -> (repo root)
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(filename: str = "params.yaml") -> Dict[str, Any]:
    """Load YAML parameters located at project root.

    Raises
    ------
    FileNotFoundError
        If *params.yaml* is not found in the expected location.
    """
    root = _project_root()
    config_path = root / filename

    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find {filename} at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]
