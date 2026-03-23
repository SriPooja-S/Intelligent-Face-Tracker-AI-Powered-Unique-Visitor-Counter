"""
config_loader.py
----------------
Loads and validates the central config.json file.
Provides a simple namespace-style accessor throughout the project.
"""

import json
import os
from pathlib import Path


class Config:
    """Wraps config.json as a nested attribute-accessible object."""

    def __init__(self, config_path: str = "config/config.json"):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self._data = json.load(f)

        # Ensure log directories exist on load
        log_dir = self._data.get("logging", {}).get("log_dir", "logs")
        for sub in ["entries", "exits", "registered"]:
            os.makedirs(os.path.join(log_dir, sub), exist_ok=True)

    def get(self, *keys, default=None):
        """Safely retrieve a nested key: config.get('detection', 'frame_skip')."""
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    # Convenient top-level section accessors
    @property
    def detection(self) -> dict:
        return self._data.get("detection", {})

    @property
    def recognition(self) -> dict:
        return self._data.get("recognition", {})

    @property
    def tracking(self) -> dict:
        return self._data.get("tracking", {})

    @property
    def logging(self) -> dict:
        return self._data.get("logging", {})

    @property
    def database(self) -> dict:
        return self._data.get("database", {})

    @property
    def display(self) -> dict:
        return self._data.get("display", {})

    @property
    def video_source(self) -> str:
        if self._data.get("use_rtsp", False):
            return self._data.get("rtsp_url", "")
        return self._data.get("video_source", "")
