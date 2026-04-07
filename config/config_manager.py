"""
配置管理器：加載、驗證、合併 YAML 配置

支援：
  1. 從 YAML 加載機器人和訓練配置
  2. 環境變數覆蓋 (e.g., LEARNING_RATE=5e-4)
  3. CLI 參數覆蓋
  4. 配置驗證和一致性檢查
  5. 配置摘要輸出
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import argparse
import logging

from .robot_config import (
    RobotConfig, PhysicsConfig, RewardConfig, TrainingConfig, 
    SACConfig, VecNormalizeConfig, CallbackConfig
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 配置路徑定義
# ═══════════════════════════════════════════════════════════════════════════════

def get_config_dir() -> Path:
    """獲取配置目錄（相對於此文件）"""
    return Path(__file__).parent.parent / "configs"


def get_robot_config_path(robot_name: str) -> Path:
    """獲取機器人配置文件路徑"""
    return get_config_dir() / "robots" / f"{robot_name}.yaml"


def get_training_config_path(config_name: str) -> Path:
    """獲取訓練配置文件路徑"""
    return get_config_dir() / "training" / f"{config_name}.yaml"


# ═══════════════════════════════════════════════════════════════════════════════
# YAML 加載/序列化
# ═══════════════════════════════════════════════════════════════════════════════

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """載入 YAML 檔案"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """保存為 YAML 檔案"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 配置構建器
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigBuilder:
    """從 YAML 和參數構建配置對象"""
    
    @staticmethod
    def build_physics(data: Dict[str, Any]) -> PhysicsConfig:
        """構建物理配置"""
        phys_data = data.get("physics", {})
        return PhysicsConfig(
            physics_hz=phys_data.get("physics_hz", 500),
            policy_hz=phys_data.get("policy_hz", 50),
            gravity=phys_data.get("gravity", -9.81),
            num_solver_iterations=phys_data.get("num_solver_iterations", 50),
        )
    
    @staticmethod
    def build_reward(data: Dict[str, Any]) -> RewardConfig:
        """構建獎勵配置"""
        reward_data = data.get("reward", {})
        return RewardConfig(
            alive_bonus=reward_data.get("alive_bonus", 2.0),
            height_reward_scale=reward_data.get("height_reward_scale", 3.0),
            forward_reward_scale=reward_data.get("forward_reward_scale", 1.5),
            smooth_penalty_scale=reward_data.get("smooth_penalty_scale", 0.001),
            death_penalty=reward_data.get("death_penalty", -30.0),
        )
    
    @staticmethod
    def build_robot_config(data: Dict[str, Any]) -> RobotConfig:
        """從 YAML 字典構建 RobotConfig"""
        cfg = RobotConfig(
            name=data.get("robot_name", "<unnamed>"),
            description=data.get("description", ""),
            robot_description_name=data.get("robot_description_name", ""),
            urdf_path=data.get("urdf_path"),
            physics=ConfigBuilder.build_physics(data),
            max_steps=data.get("max_steps", 1000),
            torso_min_z=data.get("torso_min_z", 0.60),
            spawn_z=data.get("spawn_z", 1.00),
            plane_friction=data.get("plane_friction", 0.8),
            foot_friction=data.get("foot_friction", 1.2),
            active_joints=data.get("active_joints", []),
            max_torque=data.get("max_torque", []),
            stand_pose=data.get("stand_pose", {}),
            feet_link_indices=data.get("feet_link_indices", {}),
            lock_joints=data.get("lock_joints", []),
            obs_dim=data.get("obs_dim", 0),
            action_dim=data.get("action_dim", 0),
            reward=ConfigBuilder.build_reward(data),
            metadata=data.get("metadata", {}),
        )
        return cfg
    
    @staticmethod
    def build_sac_config(data: Dict[str, Any]) -> SACConfig:
        """構建 SAC 配置"""
        sac_data = data.get("sac", {})
        return SACConfig(
            learning_rate=sac_data.get("learning_rate", 3e-4),
            buffer_size=sac_data.get("buffer_size", 1_000_000),
            batch_size=sac_data.get("batch_size", 256),
            tau=sac_data.get("tau", 0.005),
            gamma=sac_data.get("gamma", 0.99),
            train_freq=sac_data.get("train_freq", 1),
            gradient_steps=sac_data.get("gradient_steps", 1),
            net_arch=sac_data.get("net_arch", [400, 300]),
            use_sde=sac_data.get("use_sde", False),
        )
    
    @staticmethod
    def build_vecnormalize_config(data: Dict[str, Any]) -> VecNormalizeConfig:
        """構建 VecNormalize 配置"""
        vec_data = data.get("vecnormalize", {})
        return VecNormalizeConfig(
            norm_obs=vec_data.get("norm_obs", True),
            norm_reward=vec_data.get("norm_reward", True),
            clip_obs=vec_data.get("clip_obs", 10.0),
        )
    
    @staticmethod
    def build_callback_config(data: Dict[str, Any]) -> CallbackConfig:
        """構建回調配置"""
        cb_data = data.get("callbacks", {})
        return CallbackConfig(
            checkpoint_save_freq=cb_data.get("checkpoint_save_freq", 100_000),
            eval_freq=cb_data.get("eval_freq", 50_000),
            n_eval_episodes=cb_data.get("n_eval_episodes", 5),
            progress_print_freq=cb_data.get("progress_print_freq", 10_000),
        )
    
    @staticmethod
    def build_training_config(robot_cfg: RobotConfig, training_data: Dict[str, Any]) -> TrainingConfig:
        """從訓練 YAML 和機器人配置構建 TrainingConfig"""
        cfg = TrainingConfig(
            robot=robot_cfg,
            algorithm=training_data.get("algorithm", "SAC"),
            sac=ConfigBuilder.build_sac_config(training_data),
            n_envs=training_data.get("n_envs", 8),
            vecnormalize=ConfigBuilder.build_vecnormalize_config(training_data),
            total_timesteps=training_data.get("total_timesteps", 5_000_000),
            callbacks=ConfigBuilder.build_callback_config(training_data),
            model_dir=training_data.get("model_dir", "models"),
            log_dir=training_data.get("log_dir", "logs"),
            resume=training_data.get("resume", False),
            render=training_data.get("render", False),
            seed=training_data.get("seed", 42),
            use_simulation=training_data.get("use_simulation", True),
            metadata=training_data.get("metadata", {}),
        )
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 參數覆蓋
# ═══════════════════════════════════════════════════════════════════════════════

def apply_overrides(cfg: TrainingConfig, overrides: Dict[str, Any]) -> None:
    """
    將覆蓋參數應用到配置對象
    
    支援多層級覆蓋：
      - robot_max_torque: [1, 2, 3] (頂層)
      - sac__learning_rate: 5e-4 (使用雙下劃線分隔)
    """
    for key, value in overrides.items():
        if "__" in key:
            # 嵌套參數：object__field
            parts = key.split("__", 1)
            obj_name = parts[0]
            field_name = parts[1]
            
            if hasattr(cfg, obj_name):
                obj = getattr(cfg, obj_name)
                if hasattr(obj, field_name):
                    setattr(obj, field_name, value)
                else:
                    logger.warning(f"Unknown field: {obj_name}.{field_name}")
            else:
                logger.warning(f"Unknown object: {obj_name}")
        else:
            # 頂層參數
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.warning(f"Unknown config field: {key}")


def get_env_overrides() -> Dict[str, Any]:
    """
    從環境變數讀取覆蓋參數
    
    命名約定：
      PYWALKING_<PARAM_NAME>=<value>
      
    例如：
      PYWALKING_LEARNING_RATE=5e-4
      PYWALKING_SAC__BATCH_SIZE=512
      PYWALKING_N_ENVS=16
    """
    overrides = {}
    prefix = "PYWALKING_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            param_name = key[len(prefix):].lower()
            
            # 嘗試推斷類型
            try:
                # 嘗試浮點
                if '.' in value or 'e' in value.lower():
                    overrides[param_name] = float(value)
                # 嘗試整數
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    overrides[param_name] = int(value)
                # 布林值
                elif value.lower() in ('true', 'false'):
                    overrides[param_name] = value.lower() == 'true'
                # 列表 (e.g., "[1,2,3]" or "1,2,3")
                elif value.startswith('[') and value.endswith(']'):
                    import json
                    overrides[param_name] = json.loads(value)
                # 列表 (逗號分隔)
                elif ',' in value:
                    overrides[param_name] = [v.strip() for v in value.split(',')]
                # 字符串
                else:
                    overrides[param_name] = value
            except ValueError:
                overrides[param_name] = value
    
    return overrides


# ═══════════════════════════════════════════════════════════════════════════════
# 主配置加載器
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """配置管理中心"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    def load_robot_config(self, robot_name: str) -> RobotConfig:
        """加載機器人配置"""
        path = get_robot_config_path(robot_name)
        data = load_yaml(path)
        cfg = ConfigBuilder.build_robot_config(data)
        if self.verbose:
            logger.info(f"✓ Loaded robot config: {robot_name}")
        return cfg
    
    def load_training_config(self, training_config_name: str = "default") -> Dict[str, Any]:
        """加載訓練配置 (YAML data)"""
        path = get_training_config_path(training_config_name)
        data = load_yaml(path)
        if self.verbose:
            logger.info(f"✓ Loaded training config: {training_config_name}")
        return data
    
    def build_training(
        self, 
        robot_name: str, 
        training_config_name: str = "default",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> TrainingConfig:
        """
        構建完整的訓練配置
        
        Args:
            robot_name: 機器人名稱 (e.g., 'cassie')
            training_config_name: 訓練配置名稱 (e.g., 'default')
            overrides: 手動覆蓋參數
        
        Returns:
            TrainingConfig 對象
        """
        # 加載機器人配置
        robot_cfg = self.load_robot_config(robot_name)
        
        # 加載訓練配置
        training_data = self.load_training_config(training_config_name)
        
        # 構建 TrainingConfig
        cfg = ConfigBuilder.build_training_config(robot_cfg, training_data)
        
        # 應用環境變數覆蓋
        env_overrides = get_env_overrides()
        if env_overrides:
            if self.verbose:
                logger.info(f"  Applying {len(env_overrides)} environment variable overrides")
            apply_overrides(cfg, env_overrides)
        
        # 應用手動覆蓋
        if overrides:
            if self.verbose:
                logger.info(f"  Applying {len(overrides)} manual overrides")
            apply_overrides(cfg, overrides)
        
        # 驗證
        cfg.robot.__post_init__()  # 檢查一致性
        
        if self.verbose:
            logger.info(f"✓ Training config ready for: {robot_name}")
        
        return cfg
    
    def print_summary(self, cfg: TrainingConfig, max_width: int = 100) -> None:
        """列印配置摘要"""
        from pprint import pformat
        
        print("\n" + "="*max_width)
        print(f"  TRAINING CONFIGURATION SUMMARY  ({cfg.robot.name})".center(max_width))
        print("="*max_width)
        
        print(f"\n┌─ Robot: {cfg.robot.name}")
        print(f"│  Description: {cfg.robot.description}")
        print(f"│  Active Joints: {len(cfg.robot.active_joints)}")
        print(f"│  Obs Dim: {cfg.robot.obs_dim}, Action Dim: {cfg.robot.action_dim}")
        
        print(f"\n┌─ Physics")
        print(f"│  Freq: {cfg.robot.physics.physics_hz} Hz (policy: {cfg.robot.physics.policy_hz} Hz)")
        print(f"│  Substeps: {cfg.robot.physics.substeps}")
        
        print(f"\n┌─ Training")
        print(f"│  Algorithm: {cfg.algorithm}")
        print(f"│  Total Timesteps: {cfg.total_timesteps:,}")
        print(f"│  N Envs: {cfg.n_envs}")
        
        print(f"\n┌─ SAC Hyperparameters")
        print(f"│  LR: {cfg.sac.learning_rate:.0e}")
        print(f"│  Buffer: {cfg.sac.buffer_size:,}")
        print(f"│  Batch: {cfg.sac.batch_size}")
        print(f"│  Network: {cfg.sac.net_arch}")
        
        print(f"\n┌─ Reward Weights")
        print(f"│  Alive: {cfg.robot.reward.alive_bonus}")
        print(f"│  Height: {cfg.robot.reward.height_reward_scale}")
        print(f"│  Forward: {cfg.robot.reward.forward_reward_scale}")
        print(f"│  Smooth: {cfg.robot.reward.smooth_penalty_scale}")
        
        print(f"\n┌─ Paths")
        print(f"│  Models: {cfg.model_dir}/{cfg.robot.name}")
        print(f"│  Logs: {cfg.log_dir}/{cfg.robot.name}")
        
        print("\n" + "="*max_width + "\n")
