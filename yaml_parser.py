"""YAML configuration parser."""
from __future__ import annotations

from typing import Any, Dict
import yaml


class YamlParser:
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

    def get_config(self) -> Dict[str, Any]:
        return self._config
