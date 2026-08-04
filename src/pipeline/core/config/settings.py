"""Centralized configuration settings loader for the AI Quantum 2026 Pipeline.

Loads config.yaml, assets.yaml, features.yaml, model.yaml and .env centrally.
"""

import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional


def _find_root_dir() -> Path:
    """Searches upward from this file to find the workspace root containing config/."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config" / "api.yaml").exists() or (parent / "config" / "config.yaml").exists():
            return parent
    return current.parent.parent.parent


ROOT_DIR = _find_root_dir()
load_dotenv(dotenv_path=ROOT_DIR / ".env")


class ConfigurationError(Exception):
    """Raised when there is an error loading or parsing configuration."""
    pass


class ConfigNode:
    """A dot-accessible dictionary wrapper for nested config access."""

    def __init__(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    [ConfigNode(item) if isinstance(item, dict) else item for item in value],
                )
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ConfigNode back into a raw dictionary."""
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigNode) else item for item in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with a default fallback."""
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__


class GlobalSettings:
    """Centralized settings registry.

    Loads:
        - api.yaml / config.yaml: database, system, APIs
        - assets.yaml: asset class definitions
        - features.yaml: indicator definitions and pipeline_settings
        - model.yaml: model engine configurations
    """

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        if config_dir is None:
            config_dir = ROOT_DIR / "config"

        self.config_dir = config_dir
        self.root_dir = ROOT_DIR
        self.load()

    def _replace_env_variables(self, text: str) -> str:
        """Replace ${ENV_VAR} placeholders with actual environment variables."""
        pattern = re.compile(r"\$\{(\w+)\}")

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return os.getenv(var_name, f"${{{var_name}}}")

        return pattern.sub(replacer, text)

    def _load_yaml(self, path: Path, optional: bool = False) -> Dict[str, Any]:
        """Load a YAML file with environment variable interpolation."""
        if not path.exists():
            if optional:
                return {}
            raise ConfigurationError(f"Config file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        interpolated = self._replace_env_variables(raw)
        return yaml.safe_load(interpolated) or {}

    def load(self) -> None:
        """Loads and parses all configuration files."""
        main_config_path = self.config_dir / "api.yaml"
        if not main_config_path.exists():
            main_config_path = self.config_dir / "config.yaml"
        config_data = self._load_yaml(main_config_path)
        assets_data = self._load_yaml(self.config_dir / "assets.yaml")
        features_data = self._load_yaml(self.config_dir / "features.yaml", optional=True)
        model_data = self._load_yaml(self.config_dir / "model.yaml", optional=True)
        market_data = self._load_yaml(self.config_dir / "market.yaml", optional=True)

        # Merge config + assets
        merged = {**config_data, **assets_data}

        # Core settings
        self.system = ConfigNode(merged.get("system", {}))
        self.database = ConfigNode(merged.get("database", {}))
        self.apis = ConfigNode(merged.get("apis", {}))
        self.asset_class = ConfigNode(merged.get("asset_class", {}))
        self.macro_indices = ConfigNode(merged.get("macro_indices", {}))

        # Feature-specific settings
        raw_tech = features_data.get("technical_indicators", [])
        raw_macro = features_data.get("macro_indicators", [])
        for item in raw_tech:
            if "category" not in item:
                item["category"] = "trend"
        for item in raw_macro:
            item["category"] = "macro"

        self.indicators: List[Dict[str, Any]] = features_data.get("indicators", raw_tech + raw_macro)
        self.pipeline_settings = ConfigNode(features_data.get("pipeline_settings", {}))


        # Model-specific settings
        self.model_engine = ConfigNode(model_data.get("model_engine", {}))
        
        # Market microstructure settings
        self.market = ConfigNode(market_data)

    def reload(self) -> None:
        """Reloads configuration files from disk."""
        self.load()


# Global settings singleton
settings = GlobalSettings()

# Maintain backward compatibility for MODEL_CONFIG and MARKET_CONFIG
MODEL_CONFIG = settings.model_engine.to_dict() if settings.model_engine else {}
MARKET_CONFIG = settings.market.to_dict() if settings.market else {}


def reload_settings() -> GlobalSettings:
    """Reloads all YAML settings from disk and updates MODEL_CONFIG and MARKET_CONFIG in-place."""
    global settings
    settings.load()
    if settings.model_engine:
        MODEL_CONFIG.clear()
        MODEL_CONFIG.update(settings.model_engine.to_dict())
    if settings.market:
        MARKET_CONFIG.clear()
        MARKET_CONFIG.update(settings.market.to_dict())
    return settings
