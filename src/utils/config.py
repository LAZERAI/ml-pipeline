"""
Configuration Module
--------------------
Centralized configuration management using YAML.
"""

import os
import yaml
from typing import Any, Dict
from .logger import get_logger

logger = get_logger(__name__)


class Config:
    """
    Configuration manager for the ML Pipeline.
    
    Loads settings from YAML files and environment variables.
    """
    
    DEFAULT_CONFIG = {
        "data": {
            "raw_path": "data/raw",
            "processed_path": "data/processed",
            "test_size": 0.2
        },
        "model": {
            "type": "random_forest",
            "random_state": 42,
            "n_estimators": 100
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        """
        Initialize Config.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if exists
        if os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
        
        logger.info("Configuration loaded")
    
    def _load_from_file(self, path: str):
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._deep_update(self.config, file_config)
            logger.info(f"Config loaded from: {path}")
        except Exception as e:
            logger.warning(f"Could not load config from {path}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Map environment variables to config keys
        env_mapping = {
            "MODEL_TYPE": ("model", "type"),
            "MODEL_PATH": ("model", "path"),
            "API_HOST": ("api", "host"),
            "API_PORT": ("api", "port"),
            "LOG_LEVEL": ("logging", "level"),
            "TEST_SIZE": ("data", "test_size")
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested(self.config, config_path, value)
                logger.debug(f"Config override from env: {env_var}")
    
    def _deep_update(self, base: dict, update: dict):
        """Deep merge update into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _set_nested(self, config: dict, path: tuple, value: Any):
        """Set a nested config value."""
        for key in path[:-1]:
            config = config.setdefault(key, {})
        config[path[-1]] = value
    
    def get(self, *keys, default=None) -> Any:
        """
        Get a config value by nested keys.
        
        Args:
            *keys: Nested keys (e.g., 'model', 'type')
            default: Default value if not found
        
        Returns:
            Config value
        """
        result = self.config
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return default
        return result if result is not None else default
    
    def get_all(self) -> Dict:
        """Get all configuration."""
        return self.config.copy()
    
    def save(self, path: str = None):
        """
        Save current configuration to file.
        
        Args:
            path: Path to save (default: original path)
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Config saved to: {save_path}")


# Create default config file if it doesn't exist
def create_default_config():
    """Create default configuration file."""
    config_dir = "configs"
    config_path = os.path.join(config_dir, "pipeline_config.yaml")
    
    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)
        
        default_config = {
            "data": {
                "raw_path": "data/raw",
                "processed_path": "data/processed",
                "test_size": 0.2
            },
            "model": {
                "type": "random_forest",
                "random_state": 42,
                "n_estimators": 100
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default config: {config_path}")


if __name__ == "__main__":
    create_default_config()
    config = Config()
    print("Configuration loaded:")
    print(yaml.dump(config.get_all(), default_flow_style=False))
