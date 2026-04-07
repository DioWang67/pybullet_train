"""
通用雙足行走環境基類

所有機器人環境都應繼承此基類，只需實現機器人特定的钩子方法
"""

from abc import abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import RobotConfig
from simulators.robot_interface import RobotInterface


class WalkingEnv(gym.Env):
    """
    通用雙足行走環境基類
    
    設計特點：
    - 機器人無關的邏輯（獎勵、觀測、終止）
    - 抽象機械人特定操作為钩子方法
    - 支持模擬和真實硬體via RobotInterface
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        robot_config: RobotConfig,
        robot_simulator: RobotInterface,
        render_mode: Optional[str] = None,
    ):
        """
        初始化步行環境
        
        Args:
            robot_config: RobotConfig 物件 (包含所有參數)
            robot_simulator: RobotInterface 實現 (PyBullet 或真實硬體)
            render_mode: 渲染模式 ('human', 'rgb_array', None)
        """
        self.robot_config = robot_config
        self.robot_simulator = robot_simulator
        self.render_mode = render_mode
        
        # 狀態
        self._step_count = 0
        
        # 動作/觀測空間 (由 hook 實現)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(robot_config.action_dim,),
            dtype=np.float32,
        )
        
        obs_limit = np.full(robot_config.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_limit, obs_limit, dtype=np.float32)
    
    # ═════════════════════════════════════════════════════════════════════════
    # Gymnasium API (標準生命週期)
    # ═════════════════════════════════════════════════════════════════════════
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重設環境"""
        super().reset(seed=seed)
        
        # 連線 (如果尚未連線)
        if not self._is_simulator_connected():
            self.robot_simulator.connect()
        
        # 重設模擬器
        base_pos = self._get_base_position()
        base_orn = self._get_base_orientation()
        self.robot_simulator.reset(base_pos, base_orn)
        
        # 初始化機器人
        self._init_robot()
        
        # 沉降
        self._settle()
        
        # 啟用渲染
        if self.render_mode == "human":
            self.robot_simulator.enable_rendering(base_pos)
        
        self._step_count = 0
        obs = self._observe()
        
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """執行一步"""
        # 應用動作
        self._apply_action(action)
        
        # 步進模擬
        for _ in range(self.robot_config.physics.substeps):
            self.robot_simulator.step()
        
        self._step_count += 1
        obs = self._observe()
        
        # 計算獎勵和終止條件
        reward, terminated, truncated = self._compute_reward_and_termination()
        
        # 更新相機
        if self.render_mode == "human":
            robot_pos = self.robot_simulator.get_base_position()
            self.robot_simulator.update_camera(robot_pos)
        
        return obs, reward, terminated, truncated, {}
    
    def render(self):
        """渲染 (由 PyBullet 自行處理)"""
        pass
    
    def close(self):
        """關閉環境"""
        if self._is_simulator_connected():
            self.robot_simulator.disconnect()
    
    # ═════════════════════════════════════════════════════════════════════════
    # 獎勵和終止邏輯 (通用)
    # ═════════════════════════════════════════════════════════════════════════
    
    def _compute_reward_and_termination(self) -> Tuple[float, bool, bool]:
        """
        計算獎勵和終止條件（通用邏輯）
        
        Returns:
            (reward, terminated, truncated)
        """
        pos = self.robot_simulator.get_base_position()
        lin_vel = self.robot_simulator.get_base_linear_velocity()
        torques = self._get_last_torques()
        
        # 終止條件：倒下
        terminated = bool(pos[2] < self.robot_config.torso_min_z)
        truncated = bool(self._step_count >= self.robot_config.max_steps)
        
        # 獎勵分量 (使用 robot_config 中的係數)
        r = self.robot_config.reward
        
        alive_bonus = r.alive_bonus if not terminated else 0.0
        height_reward = (pos[2] - self.robot_config.torso_min_z) * r.height_reward_scale
        forward_reward = lin_vel[0] * r.forward_reward_scale
        smooth_penalty = r.smooth_penalty_scale * float(np.sum(torques ** 2))
        death_penalty = r.death_penalty if terminated else 0.0
        
        reward = (alive_bonus + height_reward + forward_reward
                  - smooth_penalty + death_penalty)

        return float(reward), terminated, truncated
    
    # ═════════════════════════════════════════════════════════════════════════
    # 抽象方法 (由子類實現) - 機器人特定邏輯
    # ═════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def _get_base_position(self) -> np.ndarray:
        """獲取初始軀幹位置 [x, y, z]"""
        pass
    
    @abstractmethod
    def _get_base_orientation(self) -> np.ndarray:
        """獲取初始軀幹方向 [qx, qy, qz, qw]"""
        pass
    
    @abstractmethod
    def _init_robot(self) -> None:
        """初始化機器人 (設置摩擦、站立姿態等)"""
        pass
    
    @abstractmethod
    def _settle(self) -> None:
        """讓機器人沉降到穩定位置 (通常 0.3 秒模擬)"""
        pass
    
    @abstractmethod
    def _observe(self) -> np.ndarray:
        """獲取觀測"""
        pass
    
    @abstractmethod
    def _apply_action(self, action: np.ndarray) -> None:
        """應用標準化動作 [-1, 1] 到機器人"""
        pass
    
    @abstractmethod
    def _get_last_torques(self) -> np.ndarray:
        """獲取上一步應用的扭矩 (用於獎勵計算)"""
        pass
    
    # ═════════════════════════════════════════════════════════════════════════
    # 輔助方法
    # ═════════════════════════════════════════════════════════════════════════
    
    def _is_simulator_connected(self) -> bool:
        """檢查模擬器是否已連線"""
        try:
            if hasattr(self.robot_simulator, 'get_client_id'):
                return self.robot_simulator.get_client_id() >= 0
            return True
        except Exception:
            return False
