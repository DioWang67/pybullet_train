"""
Configuration subsystem for PyBullet Walking

Exports:
  - RobotConfig, TrainingConfig: Main configuration dataclasses
  - PhysicsConfig, RewardConfig: Sub-configs
  - ConfigManager: Main API for loading configs
"""

from .robot_config import (
    PhysicsConfig,
    RewardConfig,
    RobotConfig,
    SACConfig,
    VecNormalizeConfig,
    CallbackConfig,
    TrainingConfig,
)

from .config_manager import (
    ConfigManager,
    ConfigBuilder,
    get_config_dir,
    get_robot_config_path,
    get_training_config_path,
    load_yaml,
    save_yaml,
)

__all__ = [
    # Configs
    "PhysicsConfig",
    "RewardConfig",
    "RobotConfig",
    "SACConfig",
    "VecNormalizeConfig",
    "CallbackConfig",
    "TrainingConfig",
    # Manager
    "ConfigManager",
    "ConfigBuilder",
    "get_config_dir",
    "get_robot_config_path",
    "get_training_config_path",
    "load_yaml",
    "save_yaml",
]
