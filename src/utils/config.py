"""Utilities for loading, saving, and updating YAML configs."""

import logging
from pathlib import Path
from typing import Any

import yaml
from box import Box


class Config:
    """Helper for YAML configs represented as ``Box`` objects.

    Keep config files as plain YAML on disk, but use ``Box`` in Python so
    callers can access nested parameters with dot notation such as
    ``cfg.simulation.training.lr``.
    """

    def __init__(
        self,
        default_box: bool = False,
    ) -> None:
        # default_box=False makes missing keys fail loudly instead of creating
        # silent empty boxes, which is safer for experiment configs.
        self.default_box = default_box
        self.logger = logging.getLogger(__name__)

    def load(self, path: str | Path) -> Box:
        """Load a YAML config as a ``Box``."""
        path = Path(path)
        self.logger.info("Loading config from %s", path)

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = Box(data, default_box=self.default_box)
        self.logger.debug("Loaded config keys: %s", list(cfg.keys()))
        return cfg

    def save(self, cfg: Box, path: str | Path) -> None:
        """Save a ``Box`` config as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving config to %s", path)

        with path.open("w", encoding="utf-8") as f:
            # Convert Box back to plain Python containers before YAML dumping.
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    def update(self, cfg: Box, updates: dict[str, Any] | None) -> Box:
        """Update ``cfg`` with a nested dictionary.

        Example:
            config.update(cfg, {"simulation": {"training": {"lr": 1e-3}}})
        """
        if updates is None:
            self.logger.debug("No config updates provided.")
            return cfg
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary or None.")

        for key, value in updates.items():
            if isinstance(value, dict):
                # Nested dictionaries are merged into existing subsections so
                # unrelated config values under the same key are preserved.
                if key not in cfg or cfg[key] is None:
                    cfg[key] = Box({}, default_box=self.default_box)
                self.update(cfg[key], value)
            else:
                cfg[key] = value

        self.logger.debug("Updated config with keys: %s", list(updates.keys()))
        return cfg
