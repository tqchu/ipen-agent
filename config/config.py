import os
import logging
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class Config:
    """
    Singleton configuration class that loads settings from .env file
    and provides centralized access to configuration values.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self._config: Dict[str, Any] = {}
        self._load_from_env()
        self._initialized = True

    def _load_from_env(self) -> None:
        """Load configuration from .env file."""
        env_path = Path('/home/truongchu/Academic/Graduation_Thesis/Project/AI/ipen-agent/.env')
        if not env_path.exists():
            self.logger.warning(f"Environment file not found: {env_path.absolute()}")
            return

        self.logger.info(f"Loading configuration from {env_path.absolute()}")
        load_dotenv(env_path)

        # Load all environment variables into config dict
        for key, value in os.environ.items():
            self._config[key] = value

        self.logger.debug(f"Loaded {len(self._config)} configuration values")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config values."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of config values."""
        self.set(key, value)

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values as dictionary."""
        return self._config.copy()