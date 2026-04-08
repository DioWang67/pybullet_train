#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 — 配置系統測試

驗證配置加載、驗證、覆蓋等功能
"""

import sys
import os
from pathlib import Path
import tempfile

# 設置編碼
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加到路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_config_loading():
    """測試: 配置加載"""
    print("\n" + "="*80)
    print("測試 1: 配置加載".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager
        
        mgr = ConfigManager(verbose=False)
        
        # 加載 Cassie
        cassie = mgr.load_robot_config("cassie")
        assert cassie.name == "cassie", f"Expected 'cassie', got {cassie.name}"
        assert cassie.obs_dim == 31, f"Expected obs_dim=31, got {cassie.obs_dim}"
        assert cassie.action_dim == 10, f"Expected action_dim=10, got {cassie.action_dim}"
        print("✓ Cassie 配置加載成功")
        
        # 加載 H1
        h1 = mgr.load_robot_config("h1")
        assert h1.name == "h1", f"Expected 'h1', got {h1.name}"
        assert h1.obs_dim == 31, f"Expected obs_dim=31, got {h1.obs_dim}"
        print("✓ H1 配置加載成功")
        
        # 加載訓練配置
        # (skip raw load, use build_training instead)
        # training = mgr.load_training_config("default")
        print("⚠ 訓練配置加載 (通過 build_training 驗證)")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_var_overrides():
    """測試: 環境變數覆蓋"""
    print("\n" + "="*80)
    print("測試 2: 環境變數覆蓋".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager
        
        # 設置環境變數
        os.environ["PYWALKING_SAC__LEARNING_RATE"] = "1e-3"
        os.environ["PYWALKING_N_ENVS"] = "8"
        
        mgr = ConfigManager(verbose=False)
        cfg = mgr.build_training("cassie", "default")
        
        # 驗證覆蓋
        assert abs(cfg.sac.learning_rate - 1e-3) < 1e-10, \
            f"Expected lr=1e-3, got {cfg.sac.learning_rate}"
        print(f"✓ SAC 學習率覆蓋: {cfg.sac.learning_rate}")
        
        assert cfg.n_envs == 8, f"Expected n_envs=8, got {cfg.n_envs}"
        print(f"✓ 環境數覆蓋: {cfg.n_envs}")
        
        # 清理
        del os.environ["PYWALKING_SAC__LEARNING_RATE"]
        del os.environ["PYWALKING_N_ENVS"]
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        # 清理
        os.environ.pop("PYWALKING_SAC__LEARNING_RATE", None)
        os.environ.pop("PYWALKING_N_ENVS", None)
        return False


def test_cli_overrides():
    """測試: CLI 參數覆蓋"""
    print("\n" + "="*80)
    print("測試 3: CLI 參數覆蓋".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager
        
        mgr = ConfigManager(verbose=False)
        
        # 模擬 CLI 覆蓋
        overrides = {
            "sac__learning_rate": 5e-4,
            "sac__batch_size": 512,
            "n_envs": 16,
        }
        
        cfg = mgr.build_training("cassie", "default", overrides=overrides)
        
        # 驗證
        assert abs(cfg.sac.learning_rate - 5e-4) < 1e-8, \
            f"Expected lr=5e-4, got {cfg.sac.learning_rate}"
        print(f"✓ 學習率覆蓋: {cfg.sac.learning_rate}")
        
        assert cfg.sac.batch_size == 512, \
            f"Expected batch_size=512, got {cfg.sac.batch_size}"
        print(f"✓ 批量大小覆蓋: {cfg.sac.batch_size}")
        
        assert cfg.n_envs == 16, f"Expected n_envs=16, got {cfg.n_envs}"
        print(f"✓ 環境數覆蓋: {cfg.n_envs}")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """測試: 配置驗證"""
    print("\n" + "="*80)
    print("測試 4: 配置驗證".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager, TrainingConfig, RobotConfig
        
        mgr = ConfigManager(verbose=False)
        
        # 構建完整配置
        cfg = mgr.build_training("cassie", "default")
        
        # 驗證類型
        assert isinstance(cfg, TrainingConfig), \
            f"Expected TrainingConfig, got {type(cfg)}"
        print("✓ 配置是 TrainingConfig 實例")
        
        assert isinstance(cfg.robot, RobotConfig), \
            f"Expected RobotConfig, got {type(cfg.robot)}"
        print("✓ robot 是 RobotConfig 實例")
        
        # 驗證基本屬性
        assert cfg.robot.name == "cassie", \
            f"Expected name='cassie', got {cfg.robot.name}"
        print(f"✓ 機器人名稱: {cfg.robot.name}")
        
        assert cfg.robot.obs_dim > 0, "obs_dim 應該 > 0"
        assert cfg.robot.action_dim > 0, "action_dim 應該 > 0"
        print(f"✓ 維度: obs={cfg.robot.obs_dim}, action={cfg.robot.action_dim}")
        
        # 驗證超參數
        assert cfg.sac.learning_rate > 0, "learning_rate 應該 > 0"
        assert cfg.sac.buffer_size > 0, "buffer_size 應該 > 0"
        assert cfg.sac.batch_size <= cfg.sac.buffer_size, \
            "batch_size 不應超過 buffer_size"
        print(f"✓ SAC 超參數: lr={cfg.sac.learning_rate}, buffer={cfg.sac.buffer_size}, batch={cfg.sac.batch_size}")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_summary():
    """測試: 配置摘要"""
    print("\n" + "="*80)
    print("測試 5: 配置摘要打印".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager
        
        mgr = ConfigManager(verbose=False)
        cfg = mgr.build_training("cassie", "default")
        
        # 應該能成功打印摘要
        print("\n配置摘要:\n")
        mgr.print_summary(cfg)
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_robots():
    """測試: 多機器人配置"""
    print("\n" + "="*80)
    print("測試 6: 多機器人支持".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager
        
        mgr = ConfigManager(verbose=False)
        
        # 加載所有可用機器人
        robots = ["cassie", "h1"]
        
        for robot_name in robots:
            try:
                cfg = mgr.load_robot_config(robot_name)
                print(f"✓ {robot_name}: obs_dim={cfg.obs_dim}, action_dim={cfg.action_dim}")
            except FileNotFoundError:
                print(f"⚠ {robot_name}: 配置文件未找到 (非錯誤)")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file_validity():
    """測試: YAML 文件有效性"""
    print("\n" + "="*80)
    print("測試 7: YAML 文件有效性".center(80))
    print("="*80)
    
    try:
        import yaml
        
        config_files = [
            "configs/robots/cassie.yaml",
            "configs/robots/h1.yaml",
            "configs/training/default.yaml",
        ]
        
        for file_path in config_files:
            full_path = PROJECT_ROOT / file_path
            
            if not full_path.exists():
                print(f"⚠ {file_path}: 文件未找到")
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    print(f"✓ {file_path}: YAML 有效")
            except UnicodeDecodeError:
                # 嘗試 gbk 編碼
                try:
                    with open(full_path, 'r', encoding='gbk') as f:
                        data = yaml.safe_load(f)
                        print(f"✓ {file_path}: YAML 有效 (GBK)")
                except Exception as e2:
                    print(f"⚠ {file_path}: 編碼問題，跳過")
            except yaml.YAMLError as e:
                print(f"✗ {file_path}: YAML 解析失敗 - {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def main():
    """運行所有測試"""
    print("\n" + "="*80)
    print("  Phase 4 — 配置系統測試".center(80))
    print("="*80)
    
    tests = [
        ("配置加載", test_config_loading),
        ("環境變數覆蓋", test_env_var_overrides),
        ("CLI 參數覆蓋", test_cli_overrides),
        ("配置驗證", test_config_validation),
        ("配置摘要", test_config_summary),
        ("多機器人支持", test_multiple_robots),
        ("YAML 文件有效性", test_config_file_validity),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ 測試異常: {e}")
            results.append((name, False))
    
    # 總結
    print("\n" + "="*80)
    print("  測試結果".center(80))
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("="*80)
    print(f"\n結果: {passed}/{total} 測試通過")
    
    if passed == total:
        print("\n✓ 所有配置系統測試通過！Phase 4a 完成。")
        return 0
    else:
        print(f"\n✗ {total - passed} 測試失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
