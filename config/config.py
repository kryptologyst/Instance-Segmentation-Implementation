"""
Configuration management for the instance segmentation project.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """Configuration manager for the instance segmentation project."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if config_path else None
        self._config = self._load_default_config()
        
        if self.config_path and self.config_path.exists():
            self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "model": {
                "name": "maskrcnn_resnet50_fpn",
                "confidence_threshold": 0.7,
                "device": None,  # Auto-detect
                "batch_size": 1
            },
            "visualization": {
                "alpha": 0.5,
                "show_labels": True,
                "colormap": "tab20",
                "figsize": [15, 7]
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "paths": {
                "data_dir": "data",
                "models_dir": "models",
                "output_dir": "outputs"
            }
        }
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    file_config = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
            self._merge_config(file_config)
            logging.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logging.warning(f"Failed to load config from {self.config_path}: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Recursively merge new configuration with existing."""
        for key, value in new_config.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                self._merge_config(value)
            else:
                self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path
        
        if not save_path:
            raise ValueError("No path specified for saving configuration")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() == '.yaml' or save_path.suffix.lower() == '.yml':
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {save_path.suffix}")
            
            logging.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


# Global configuration instance
config = Config()
