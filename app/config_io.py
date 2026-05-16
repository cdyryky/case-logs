from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR


def load_json_yaml(path: str | Path) -> Any:
    """Load config written as JSON-compatible YAML.

    The repository intentionally keeps YAML files JSON-compatible so core logic
    does not require PyYAML. JSON is valid YAML, and these files stay readable.
    """
    return json.loads(Path(path).read_text())


def load_resident_profile(path: str | Path | None = None) -> dict[str, Any]:
    return load_json_yaml(path or CONFIG_DIR / "resident_profile.yaml")


def load_dropdowns(path: str | Path | None = None) -> dict[str, Any]:
    return load_json_yaml(path or CONFIG_DIR / "acgme_dropdowns.yaml")

