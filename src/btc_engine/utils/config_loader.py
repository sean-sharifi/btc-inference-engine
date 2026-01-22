"""Configuration loader utility"""

from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from btc_engine.utils.constants import CONFIG_DIR

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment"""
    
    # API Keys
    deribit_api_key: str = ""
    deribit_api_secret: str = ""
    glassnode_api_key: str = ""
    
    # Database
    database_path: str = "./data/btc_engine.duckdb"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/btc_engine.log"
    
    # Pipeline
    enable_prefect: bool = True
    incremental_mode: bool = True
    
    # Dashboard
    dashboard_port: int = 8501
    dashboard_host: str = "localhost"
    
    # Demo mode
    demo_mode: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """Load YAML configuration file
    
    Args:
        config_name: Name of config file (without .yaml extension)
        
    Returns:
        Dictionary with configuration
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# Singleton settings instance
settings = Settings()
