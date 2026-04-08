#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 — 環境介面測試

驗證環境 API 和抽象層正確性 (不需要 PyBullet)
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod

# 設置編碼
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加到路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_robot_interface_exists():
    """測試: RobotInterface ABC 存在且完整"""
    print("\n" + "="*80)
    print("測試 1: RobotInterface 結構".center(80))
    print("="*80)
    
    try:
        from simulators.robot_interface import RobotInterface
        
        # 驗證是 ABC
        assert hasattr(RobotInterface, "__abstractmethods__"), \
            "RobotInterface 應該是抽象基類"
        print("✓ RobotInterface 是抽象基類")
        
        # 預期的虛擬方法
        expected_methods = [
            "connect",
            "disconnect",
            "reset",
            "apply_action",
            "step",
            "get_base_position",
            "get_base_orientation_euler",
            "get_base_linear_velocity",
            "get_base_angular_velocity",
            "get_joint_positions",
            "get_joint_velocities",
            "get_foot_contact",
        ]
        
        for method_name in expected_methods:
            assert hasattr(RobotInterface, method_name), \
                f"缺少方法: {method_name}"
        print(f"✓ 所有 {len(expected_methods)} 個虛擬方法已定義")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pybullet_simulator_exists():
    """測試: PyBulletRobotSimulator 實現"""
    print("\n" + "="*80)
    print("測試 2: PyBulletRobotSimulator 實現".center(80))
    print("="*80)
    
    try:
        from simulators.robot_interface import PyBulletRobotSimulator, RobotInterface
        
        # 驗證繼承和實現
        assert issubclass(PyBulletRobotSimulator, RobotInterface), \
            "PyBulletRobotSimulator 應該繼承 RobotInterface"
        print("✓ PyBulletRobotSimulator 繼承 RobotInterface")
        
        # 驗證初始化
        simulator = PyBulletRobotSimulator(
            robot_description="test",
            physics_hz=500,
            render_mode=None,
        )
        print("✓ PyBulletRobotSimulator 可初始化")
        
        # 驗證基本屬性
        assert hasattr(simulator, "robot_description"), "缺少 robot_description"
        assert hasattr(simulator, "physics_hz"), "缺少 physics_hz"
        print("✓ 必要屬性已存在")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_walking_env_exists():
    """測試: WalkingEnv 基類存在"""
    print("\n" + "="*80)
    print("測試 3: WalkingEnv 基類".center(80))
    print("="*80)
    
    try:
        from envs.base_walking_env import WalkingEnv
        import gymnasium as gym
        
        # 驗證繼承
        assert issubclass(WalkingEnv, gym.Env), \
            "WalkingEnv 應該繼承 gym.Env"
        print("✓ WalkingEnv 繼承 gym.Env")
        
        # 驗證是抽象基類
        assert hasattr(WalkingEnv, "__abstractmethods__"), \
            "WalkingEnv 應該是抽象的"
        print("✓ WalkingEnv 是抽象基類")
        
        # 預期的虛擬方法
        abstract_methods = [
            "_get_base_position",
            "_get_base_orientation",
            "_init_robot",
            "_observe",
            "_apply_action",
            "_settle",
            "_get_last_torques",
        ]
        
        for method_name in abstract_methods:
            assert hasattr(WalkingEnv, method_name), \
                f"缺少方法: {method_name}"
        print(f"✓ 所有 {len(abstract_methods)} 個虛擬方法已定義")
        
        # 驗證通用方法
        assert hasattr(WalkingEnv, "reset"), "缺少 reset"
        assert hasattr(WalkingEnv, "step"), "缺少 step"
        assert hasattr(WalkingEnv, "render"), "缺少 render"
        assert hasattr(WalkingEnv, "close"), "缺少 close"
        print("✓ 所有通用 Gymnasium 方法已實現")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_subclasses():
    """測試: CassieEnv 和 H1Env 正確性"""
    print("\n" + "="*80)
    print("測試 4: 環境子類結構".center(80))
    print("="*80)
    
    try:
        # 檢查結構而不導入 (避免 PyBullet)
        import inspect
        from pathlib import Path
        
        # 讀取文件以驗證結構
        cassie_file = PROJECT_ROOT / "envs" / "cassie_env.py"
        h1_file = PROJECT_ROOT / "envs" / "h1_env.py"
        
        assert cassie_file.exists(), "cassie_env.py 不存在"
        assert h1_file.exists(), "h1_env.py 不存在"
        print("✓ 環境文件存在")
        
        # 驗證基本結構 (通過文本搜索)
        with open(cassie_file, 'r') as f:
            cassie_content = f.read()
        
        assert "class CassieEnv" in cassie_content, "缺少 CassieEnv 類"
        assert "WalkingEnv" in cassie_content, "CassieEnv 應該繼承 WalkingEnv"
        assert "def _observe" in cassie_content, "缺少 _observe 方法"
        assert "def _apply_action" in cassie_content, "缺少 _apply_action 方法"
        print("✓ CassieEnv 結構完整")
        
        with open(h1_file, 'r') as f:
            h1_content = f.read()
        
        assert "class H1Env" in h1_content, "缺少 H1Env 類"
        assert "WalkingEnv" in h1_content, "H1Env 應該繼承 WalkingEnv"
        print("✓ H1Env 結構完整")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_space():
    """測試: 觀測空間結構"""
    print("\n" + "="*80)
    print("測試 5: 觀測空間定義".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager
        import numpy as np
        
        mgr = ConfigManager(verbose=False)
        
        # 檢查 Cassie 配置
        cassie_cfg = mgr.load_robot_config("cassie")
        h1_cfg = mgr.load_robot_config("h1")
        
        # 驗證觀測維度計算
        # obs_dim = 9 (base) + joints*2 + 2 (contacts)
        expected_cassie = 9 + len(cassie_cfg.active_joints) * 2 + 2
        expected_h1 = 9 + len(h1_cfg.active_joints) * 2 + 2
        
        assert cassie_cfg.obs_dim == expected_cassie, \
            f"Cassie obs_dim 計算錯誤: expected {expected_cassie}, got {cassie_cfg.obs_dim}"
        print(f"✓ Cassie 觀測維度: {cassie_cfg.obs_dim}")
        
        assert h1_cfg.obs_dim == expected_h1, \
            f"H1 obs_dim 計算錯誤: expected {expected_h1}, got {h1_cfg.obs_dim}"
        print(f"✓ H1 觀測維度: {h1_cfg.obs_dim}")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_space():
    """測試: 動作空間定義"""
    print("\n" + "="*80)
    print("測試 6: 動作空間定義".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager
        
        mgr = ConfigManager(verbose=False)
        
        cassie_cfg = mgr.load_robot_config("cassie")
        h1_cfg = mgr.load_robot_config("h1")
        
        # 動作維度應該等於活躍關節數
        assert cassie_cfg.action_dim == len(cassie_cfg.active_joints), \
            f"Cassie action_dim 不匹配: {cassie_cfg.action_dim} vs {len(cassie_cfg.active_joints)}"
        print(f"✓ Cassie 動作維度: {cassie_cfg.action_dim} (等於活躍關節數)")
        
        assert h1_cfg.action_dim == len(h1_cfg.active_joints), \
            f"H1 action_dim 不匹配: {h1_cfg.action_dim} vs {len(h1_cfg.active_joints)}"
        print(f"✓ H1 動作維度: {h1_cfg.action_dim}")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_config():
    """測試: 獎勵配置結構"""
    print("\n" + "="*80)
    print("測試 7: 獎勵配置".center(80))
    print("="*80)
    
    try:
        from config import ConfigManager, RewardConfig
        
        mgr = ConfigManager(verbose=False)
        
        cassie_cfg = mgr.load_robot_config("cassie")
        
        # 驗證獎勵配置存在
        assert hasattr(cassie_cfg, "reward_config"), "缺少 reward_config"
        assert isinstance(cassie_cfg.reward_config, RewardConfig), \
            "reward_config 應該是 RewardConfig 實例"
        print("✓ 獎勵配置存在且類型正確")
        
        # 驗證關鍵獎勵參數
        reward = cassie_cfg.reward_config
        assert hasattr(reward, "alive_bonus"), "缺少 alive_bonus"
        assert hasattr(reward, "forward_reward_scale"), "缺少 forward_reward_scale"
        assert hasattr(reward, "height_reward_scale"), "缺少 height_reward_scale"
        assert hasattr(reward, "smooth_penalty_scale"), "缺少 smooth_penalty_scale"
        assert hasattr(reward, "death_penalty"), "缺少 death_penalty"
        print("✓ 所有獎勵參數已定義")
        
        # 驗證值的合理性
        assert reward.alive_bonus > 0, "alive_bonus 應該 > 0"
        assert reward.forward_reward_scale > 0, "forward_reward_scale 應該 > 0"
        assert reward.height_reward_scale > 0, "height_reward_scale 應該 > 0"
        assert reward.smooth_penalty_scale >= 0, "smooth_penalty_scale 應該 >= 0"
        assert reward.death_penalty < 0, "death_penalty 應該 < 0"
        print("✓ 獎勵參數值合理")
        
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """運行所有測試"""
    print("\n" + "="*80)
    print("  Phase 4 — 環境介面測試".center(80))
    print("="*80)
    
    tests = [
        ("RobotInterface 結構", test_robot_interface_exists),
        ("PyBulletRobotSimulator 實現", test_pybullet_simulator_exists),
        ("WalkingEnv 基類", test_walking_env_exists),
        ("環境子類結構", test_env_subclasses),
        ("觀測空間定義", test_observation_space),
        ("動作空間定義", test_action_space),
        ("獎勵配置", test_reward_config),
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
        print("\n✓ 所有環境介面測試通過！Phase 4b 完成。")
        return 0
    else:
        print(f"\n✗ {total - passed} 測試失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
