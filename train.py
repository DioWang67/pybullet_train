#!/usr/bin/env python3
"""
统一训练脚本

取代所有旧的 train_*.py，支持任何机器人配置

使用方式：
  python train.py --robot cassie                          # 默认配置
  python train.py --robot h1 --config default            # 指定配置
  python train.py --robot cassie --resume                # 恢复训练
  python train.py --robot h1 --render                    # 带渲染训练
  python train.py --robot cassie --sac__learning_rate 5e-4  # CLI 覆盖参数
  
环境变量:
  PYWALKING_SAC__LEARNING_RATE=5e-4 python train.py --robot cassie
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ConfigManager, TrainingConfig
from callbacks import SyncVecNormCallback, TrainingProgressCallback


# ═══════════════════════════════════════════════════════════════════════════════
# 環境工廠
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(robot_name: str, render: bool = False):
    """
    動態建立環境
    
    Args:
        robot_name: 機器人名稱 ('cassie', 'h1', etc.)
        render: 是否渲染
    
    Returns:
        Gymnasium environment
    """
    from config import ConfigManager
    
    mgr = ConfigManager()
    robot_config = mgr.load_robot_config(robot_name)
    
    # 動態導入環境
    if robot_name == "cassie":
        from envs.cassie_env import CassieEnv
        env = CassieEnv(robot_config=robot_config, render_mode="human" if render else None)
    elif robot_name == "h1":
        from envs.h1_env import H1Env
        env = H1Env(robot_config=robot_config, render_mode="human" if render else None)
    else:
        raise ValueError(f"Unknown robot: {robot_name}")
    
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 參數解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="統一的雙足行走 RL 訓練腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py --robot cassie                        # 默認配置
  python train.py --robot h1 --resume                  # 繼續訓練
  python train.py --robot cassie --render              # 帶渲染
  python train.py --robot cassie --sac__learning_rate 5e-4

也可用環境變數:
  PYWALKING_SAC__LEARNING_RATE=5e-4 python train.py --robot cassie
        """,
    )
    
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        help="機器人名稱: cassie, h1, etc.",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="訓練配置名稱 (default: default)",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="從上一個檢查點恢復訓練",
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="訓練時渲染環境",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細輸出",
    )
    
    # 動態參數覆蓋 (e.g., --sac__learning_rate 5e-4)
    # 這些會被捕獲為其他參數
    parser.add_argument(
        "overrides",
        nargs="*",
        help="參數覆蓋 (e.g., sac__learning_rate=5e-4 n_envs=16)",
    )
    
    return parser.parse_args()


def parse_cli_overrides(override_strs: List[str]) -> Dict[str, any]:
    """
    解析 CLI 參數覆蓋
    
    例子:
      sac__learning_rate=5e-4
      n_envs=16
      sac__batch_size=512
    """
    overrides = {}
    for s in override_strs:
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # 嘗試推斷類型
        try:
            if value.lower() in ("true", "false"):
                overrides[key] = value.lower() == "true"
            elif "." in value or "e" in value.lower():
                overrides[key] = float(value)
            elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                overrides[key] = int(value)
            else:
                overrides[key] = value
        except ValueError:
            overrides[key] = value
    
    return overrides


# ═══════════════════════════════════════════════════════════════════════════════
# 主訓練函數
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    config: TrainingConfig,
    resume: bool = False,
    render: bool = False,
) -> None:
    """
    執行訓練
    
    Args:
        config: TrainingConfig 物件
        resume: 是否恢復訓練
        render: 是否渲染
    """
    robot_name = config.robot.name
    
    # 根據配置建立目錄
    model_dir = Path(config.model_dir) / robot_name
    log_dir = Path(config.log_dir) / robot_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"{robot_name}_sac"
    vecnorm_path = model_dir / f"{robot_name}_vecnorm.pkl"
    
    print(f"\n{'='*80}")
    print(f"  訓練開始: {robot_name}".center(80))
    print(f"{'='*80}\n")
    
    # 檢查環境
    print("檢查環境...")
    test_env = make_env(robot_name, render=False)
    check_env(test_env, warn=True)
    test_env.close()
    print("環境檢查完成。\n")
    
    # 建立向量化環境
    print(f"建立 {config.n_envs} 個並行訓練環境...")
    vec_env = DummyVecEnv([lambda: Monitor(make_env(robot_name, render=False))] * config.n_envs)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=config.vecnormalize.norm_obs,
        norm_reward=config.vecnormalize.norm_reward,
        clip_obs=config.vecnormalize.clip_obs,
    )
    
    print(f"建立 1 個評估環境...")
    eval_vec = DummyVecEnv([lambda: Monitor(make_env(robot_name, render=render))])
    eval_vec = VecNormalize(
        eval_vec,
        norm_obs=config.vecnormalize.norm_obs,
        norm_reward=False,  # 評估時不規範化獎勵
        clip_obs=config.vecnormalize.clip_obs,
        training=False,
    )
    
    print(f"✓ 訓練環境 OK (obs_dim={config.robot.obs_dim}, action_dim={config.robot.action_dim})\n")
    
    # 建立或恢復模型
    if resume and model_path.exists() and vecnorm_path.exists():
        print(f"從 {model_path}.zip 恢復訓練...\n")
        vec_env = VecNormalize.load(vecnorm_path, vec_env.venv)
        vec_env.training = True
        model = SAC.load(
            model_path,
            env=vec_env,
            custom_objects={
                "learning_rate": config.sac.learning_rate,
                "lr_schedule": lambda _: config.sac.learning_rate,
            },
        )
        start_timesteps = model.num_timesteps
    else:
        print(f"建立新 SAC 模型...")
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=config.sac.learning_rate,
            buffer_size=config.sac.buffer_size,
            batch_size=config.sac.batch_size,
            tau=config.sac.tau,
            gamma=config.sac.gamma,
            train_freq=config.sac.train_freq,
            gradient_steps=config.sac.gradient_steps,
            policy_kwargs={"net_arch": config.sac.net_arch},
            verbose=1,
        )
        start_timesteps = 0
    
    # 建立回調
    callbacks = CallbackList([
        SyncVecNormCallback(train_env=vec_env, eval_env=eval_vec),
        TrainingProgressCallback(total_timesteps=config.total_timesteps),
        CheckpointCallback(
            save_freq=config.callbacks.checkpoint_save_freq,
            save_path=str(model_dir),
            name_prefix=robot_name,
            verbose=0,
        ),
        EvalCallback(
            eval_vec,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=config.callbacks.eval_freq,
            n_eval_episodes=config.callbacks.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ),
    ])
    
    # 打印配置摘要
    from config import ConfigManager
    mgr = ConfigManager()
    mgr.print_summary(config)
    
    # 訓練
    print(f"訓練 {config.total_timesteps:,} 步，使用 {config.n_envs} 個環境...\n")
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not resume,
    )
    
    # 保存最終模型
    print(f"\n保存最終模型到 {model_path}.zip ...")
    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))
    
    print(f"✓ 訓練完成！")
    print(f"  模型: {model_path}.zip")
    print(f"  日誌: {log_dir}")
    
    # 清理
    vec_env.close()
    eval_vec.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 進入點
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """主進入點"""
    args = parse_args()
    
    # 構建配置
    mgr = ConfigManager(verbose=args.verbose)
    
    # 解析 CLI 覆蓋
    cli_overrides = parse_cli_overrides(args.overrides)
    
    try:
        config = mgr.build_training(
            robot_name=args.robot,
            training_config_name=args.config,
            overrides=cli_overrides,
        )
    except FileNotFoundError as e:
        print(f"\n❌ 配置加載失敗: {e}\n")
        sys.exit(1)
    
    # 訓練
    try:
        train(config, resume=args.resume, render=args.render)
    except KeyboardInterrupt:
        print("\n\n訓練被用戶中斷。")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 訓練失敗: {e}\n")
        raise


if __name__ == "__main__":
    main()
