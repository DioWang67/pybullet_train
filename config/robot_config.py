"""
通用機器人和訓練配置 Schema

定義所有模擬和訓練參數的統一結構體，支援從 YAML/JSON 載入и覆蓋
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 物理模擬配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsConfig:
    """物理引擎參數"""
    physics_hz: int = 500
    """物理模擬頻率 (Hz)"""
    
    policy_hz: int = 50
    """策略/控制頻率 (Hz)"""
    
    gravity: float = -9.81
    """重力加速度 (m/s²)"""
    
    num_solver_iterations: int = 50
    """PyBullet 求解器迭代次數"""
    
    @property
    def substeps(self) -> int:
        """計算每個策略步的子步數"""
        return self.physics_hz // self.policy_hz


# ═══════════════════════════════════════════════════════════════════════════════
# 獎勵函數配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JointRoles:
    """Semantic joint role → positional index within active_joints.

    This decouples the balance controller from joint index layout, enabling
    reuse across different biped morphologies (H1, Cassie, custom robots).
    All indices refer to the *position within active_joints list*, not
    the underlying URDF joint index.

    Set to None for joints the robot does not have (e.g. a robot with no
    ankle roll). The balance controller silently skips None roles.
    """
    left_hip_roll: Optional[int] = None
    left_hip_pitch: Optional[int] = None
    left_knee: Optional[int] = None
    left_ankle: Optional[int] = None
    right_hip_roll: Optional[int] = None
    right_hip_pitch: Optional[int] = None
    right_knee: Optional[int] = None
    right_ankle: Optional[int] = None

    def validate(self, n_active: int) -> None:
        """Ensure all defined roles are within range of active_joints."""
        for name, idx in asdict(self).items():
            if idx is None:
                continue
            if not (0 <= idx < n_active):
                raise ValueError(
                    f"JointRoles.{name}={idx} out of range "
                    f"for active_joints length {n_active}"
                )


@dataclass
class BalanceControlConfig:
    """Stand PD + residual-torque controller parameters.

    The low-level controller drives `stand_pose` with a PD loop and injects
    pitch/roll feedback through hip and ankle joints (distributed by the
    `*_gain` weights). The RL policy adds a residual torque on top, scaled
    by `residual_torque_scale * max_torque`.

    All parameters here are morphology-independent. Per-robot tuning lives
    in the robot YAML.
    """
    # Joint-space PD holding the stand pose
    stand_kp: float = 120.0
    stand_kd: float = 8.0

    # Residual torque budget for RL (fraction of max_torque)
    residual_torque_scale: float = 0.35

    # Body-pitch feedback (drives hip_pitch + ankle on both legs)
    pitch_kp: float = 45.0
    pitch_kd: float = 8.0
    pitch_hip_gain: float = 0.25
    pitch_ankle_gain: float = 0.35

    # Body-roll feedback (drives hip_roll differentially)
    roll_kp: float = 10.0
    roll_kd: float = 2.0
    roll_hip_gain: float = 0.5


@dataclass
class RewardConfig:
    """獎勵函數係數"""
    alive_bonus: float = 2.0
    """每步存活獎勵"""
    
    height_reward_scale: float = 3.0
    """身高獎勵係數: reward = (z - min_z) * scale"""
    
    forward_reward_scale: float = 1.5
    """前進速度獎勵係數: reward = vx * scale"""
    
    smooth_penalty_scale: float = 0.001
    """扭矩平滑懲罰係數: penalty = scale * ||torque||²"""

    posture_penalty_scale: float = 0.0
    """姿態誤差懲罰: penalty = scale * (pitch^2 + roll^2)"""

    vertical_velocity_penalty_scale: float = 0.0
    """垂直速度懲罰: penalty = scale * vz^2"""

    angular_velocity_penalty_scale: float = 0.0
    """角速度懲罰: penalty = scale * ||omega||^2"""

    pitch_termination: float = float("inf")
    """pitch 終止門檻 (rad)"""

    roll_termination: float = float("inf")
    """roll 終止門檻 (rad)"""

    death_penalty: float = -30.0
    """倒下時的終止懲罰"""


# ═══════════════════════════════════════════════════════════════════════════════
# 機器人特定配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RobotConfig:
    """單個機器人的完整配置"""
    
    # ── 基本資訊 ────────────────────────────────────────────────────────────
    name: str
    """機器人名稱 (e.g., 'cassie', 'h1', 'your_robot')"""
    
    description: str = ""
    """機器人簡述 (optional)"""
    
    robot_description_name: str = ""
    """robot_descriptions 套件中的名稱 (e.g., 'cassie_description', 'h1_description')
    如果為空，認為是高階 URDF 路徑"""
    
    urdf_path: Optional[str] = None
    """URDF 檔案路徑 (若不使用 robot_descriptions)"""
    
    # ── 環境參數 ────────────────────────────────────────────────────────────
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    """物理引擎參數"""
    
    max_steps: int = 1000
    """每個 episode 的最大步數"""
    
    torso_min_z: float = 0.60
    """身體最低高度 (m)，低於此視為倒下"""
    
    spawn_z: float = 1.00
    """初始生成高度 (m)"""
    
    plane_friction: float = 0.8
    """地面側向摩擦係數"""
    
    foot_friction: float = 1.2
    """腳底側向摩擦係數"""
    
    # ── 機器人結構 ────────────────────────────────────────────────────────────
    active_joints: List[int] = field(default_factory=list)
    """控制關節的索引列表 (PyBullet joint indices)"""
    
    max_torque: List[float] = field(default_factory=list)
    """對應控制關節的最大扭矩 (Nm)
    必須與 active_joints 長度相同"""
    
    stand_pose: Dict[int, float] = field(default_factory=dict)
    """站立初始姿態: {joint_idx: angle_rad}"""
    
    feet_link_indices: Dict[str, int] = field(default_factory=dict)
    """腳部連結索引: {'left': idx, 'right': idx}
    用於接觸檢測"""
    
    lock_joints: List[int] = field(default_factory=list)
    """需要鎖定的關節 (e.g., 手臂) 為空表示無"""

    joint_roles: JointRoles = field(default_factory=JointRoles)
    """語意關節角色映射 (生成 balance controller 用，跨機器人共享)"""

    balance_control: BalanceControlConfig = field(default_factory=BalanceControlConfig)
    """Stand PD + residual-torque 控制器參數"""

    # ── 觀測/動作空間 ────────────────────────────────────────────────────────
    obs_dim: int = 0
    """觀測空間維度 (0 → 由 __post_init__ 從 active_joints 自動推算)"""

    action_dim: int = 0
    """動作空間維度 (0 → 由 __post_init__ 設為 len(active_joints))"""

    # ── 獎勵配置 ────────────────────────────────────────────────────────────
    reward: RewardConfig = field(default_factory=RewardConfig)
    """獎勵函數係數"""
    
    # ── 額外參數 ────────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)
    """任意額外參數 (e.g., 機器人質量、廠商資訊等)"""
    
    def __post_init__(self):
        """驗證配置的一致性"""
        if not self.name:
            raise ValueError("robot_name is required")
        
        if self.action_dim == 0:
            self.action_dim = len(self.active_joints)
        
        if self.max_torque and len(self.max_torque) != len(self.active_joints):
            raise ValueError(
                f"max_torque length ({len(self.max_torque)}) must match "
                f"active_joints length ({len(self.active_joints)})"
            )
        
        if self.obs_dim == 0:
            # Default computation: base(9) + joint_angles(n) + joint_vels(m) + contacts(2)
            num_joints = len(self.active_joints) if self.active_joints else 0
            self.obs_dim = 9 + num_joints + num_joints + 2

        # Validate joint roles against active_joints length.
        if self.active_joints:
            self.joint_roles.validate(len(self.active_joints))
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典 (便於 YAML export)"""
        d = asdict(self)
        # 將 numpy array 轉為 list
        if isinstance(d.get("physics"), dict):
            pass  # 已是 dict
        if "max_torque" in d and isinstance(d["max_torque"], np.ndarray):
            d["max_torque"] = d["max_torque"].tolist()
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# 訓練配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SACConfig:
    """Soft Actor-Critic 算法超參數"""
    learning_rate: float = 3e-4
    """策略和價值網路的學習率"""
    
    buffer_size: int = 1_000_000
    """回放緩衝區大小"""
    
    batch_size: int = 256
    """訓練批次大小"""
    
    tau: float = 0.005
    """目標網路軟更新係數 (Polyak averaging)"""
    
    gamma: float = 0.99
    """折扣因子"""
    
    train_freq: int = 1
    """每次環境步後的訓練次數"""
    
    gradient_steps: int = 1
    """在每個 train_freq 中執行的梯度步數"""
    
    net_arch: List[int] = field(default_factory=lambda: [400, 300])
    """MLP 隱層大小 (e.g., [256, 256])"""
    
    use_sde: bool = False
    """使用隨機微分方程 (State-Dependent Exploration)"""


@dataclass
class VecNormalizeConfig:
    """向量化環境歸一化參數"""
    norm_obs: bool = True
    """歸一化觀測"""
    
    norm_reward: bool = True
    """歸一化獎勵（訓練時有效）"""
    
    clip_obs: float = 10.0
    """觀測值裁剪範圍 [-clip_obs, +clip_obs]"""


@dataclass
class CallbackConfig:
    """訓練回調參數"""
    checkpoint_save_freq: int = 100_000
    """檢查點保存頻率 (timesteps)"""
    
    eval_freq: int = 50_000
    """評估頻率 (timesteps)"""
    
    n_eval_episodes: int = 5
    """每次評估的 episode 數"""
    
    progress_print_freq: int = 10_000
    """進度列印頻率 (timesteps)"""


@dataclass
class TrainingConfig:
    """訓練任務配置"""
    robot: RobotConfig
    """要訓練的機器人配置"""
    
    # ── RL 算法 ──────────────────────────────────────────────
    algorithm: str = "SAC"
    """使用的 RL 算法 ('SAC', 'PPO', etc.)"""
    
    sac: SACConfig = field(default_factory=SACConfig)
    """SAC 超參數"""
    
    # ── 環境 ──────────────────────────────────────────────────
    n_envs: int = 8
    """並行環境數"""
    
    vecnormalize: VecNormalizeConfig = field(default_factory=VecNormalizeConfig)
    """向量化歸一化參數"""
    
    # ── 訓練循環 ──────────────────────────────────────────────
    total_timesteps: int = 5_000_000
    """訓練總步數"""
    
    # ── 回調 ──────────────────────────────────────────────────
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    """回調參數"""
    
    # ── 路徑 ──────────────────────────────────────────────────
    model_dir: str = "models"
    """模型保存目錄"""
    
    log_dir: str = "logs"
    """日誌目錄"""
    
    # ── 其他 ──────────────────────────────────────────────────
    resume: bool = False
    """是否從檢查點恢復訓練"""
    
    render: bool = False
    """是否渲染環境"""
    
    seed: int = 42
    """隨機種子"""
    
    use_simulation: bool = True
    """使用模擬環境 (True) 或真實硬體 (False)"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """任意額外訓練元資料"""
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return asdict(self)
