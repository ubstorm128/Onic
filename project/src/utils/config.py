"""
Configuration loading and merging utilities.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(
    data_config_path: str | Path = "configs/data.yaml",
    model_config_path: str | Path = "configs/model.yaml",
    experiment_config_path: str | Path = "configs/experiment.yaml",
) -> Dict[str, Any]:
    """Load and merge data, model, and experiment configurations."""
    config: Dict[str, Any] = {}
    config["data"] = load_yaml(data_config_path)
    config["model"] = load_yaml(model_config_path)
    config["experiment"] = load_yaml(experiment_config_path)
    return config
