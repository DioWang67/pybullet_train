"""
通用機器人控制介面

定義 PyBullet 模擬和真實硬體的統一接口，使 Environment 無關底層實現
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class RobotInterface(ABC):
    """
    通用機器人控制介面
    
    所有機器人（PyBullet、真實硬體等）都應實現此介面
    """
    
    @abstractmethod
    def connect(self) -> None:
        """連線到機器人/模擬器"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """斷開連線"""
        pass
    
    @abstractmethod
    def reset(self, base_pos: np.ndarray, base_orn: np.ndarray) -> None:
        """
        重設機器人 base 位置/方向/速度。

        重要合約：此方法 **不重置關節位置/速度**。
        子類別的環境 (`WalkingEnv._init_robot`) 必須在 `reset()` 後
        呼叫 `reset_joint_state(stand_pose)` 才能保證 episode 起點乾淨。
        若忘記呼叫，前一個 episode 的關節速度會直接帶到下一個 episode。

        Args:
            base_pos: 軀幹位置 [x, y, z]
            base_orn: 軀幹方向 [qx, qy, qz, qw] (quaternion)
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """執行一個模擬/控制步"""
        pass
    
    # ── 觀測 ──────────────────────────────────────────────────────────────
    
    @abstractmethod
    def get_base_position(self) -> np.ndarray:
        """獲取軀幹位置 [x, y, z]"""
        pass
    
    @abstractmethod
    def get_base_orientation_euler(self) -> np.ndarray:
        """獲取軀幹方向 (Euler 角) [roll, pitch, yaw]"""
        pass
    
    @abstractmethod
    def get_base_linear_velocity(self) -> np.ndarray:
        """獲取軀幹線速度 [vx, vy, vz]"""
        pass
    
    @abstractmethod
    def get_base_angular_velocity(self) -> np.ndarray:
        """獲取軀幹角速度 [wx, wy, wz]"""
        pass
    
    @abstractmethod
    def get_joint_positions(self, joint_indices: list) -> np.ndarray:
        """
        獲取指定關節的位置角度
        
        Args:
            joint_indices: 關節索引列表
        
        Returns:
            np.ndarray: 對應關節的角度 [rad]
        """
        pass
    
    @abstractmethod
    def get_joint_velocities(self, joint_indices: list) -> np.ndarray:
        """
        獲取指定關節的速度
        
        Args:
            joint_indices: 關節索引列表
        
        Returns:
            np.ndarray: 對應關節的速度 [rad/s]
        """
        pass
    
    @abstractmethod
    def get_foot_contact(self, foot_link_indices: Dict[str, int]) -> Dict[str, bool]:
        """
        獲取腳部接觸狀態
        
        Args:
            foot_link_indices: 腳部 link 索引字典
                               {'left': idx_left, 'right': idx_right}
        
        Returns:
            {'left': bool, 'right': bool} - 是否接觸地面
        """
        pass
    
    # ── 設置 ──────────────────────────────────────────────────────────────
    
    @abstractmethod
    def set_dynamics(
        self,
        link_index: int,
        mass: Optional[float] = None,
        lateral_friction: Optional[float] = None,
        restitution: Optional[float] = None,
    ) -> None:
        """
        設置 link 的動力學參數
        
        Args:
            link_index: Link 索引 (-1 表示軀幹)
            mass: 質量 (kg)
            lateral_friction: 側向摩擦係數
            restitution: 恢復係數
        """
        pass
    
    # ── 渲染 ──────────────────────────────────────────────────────────────
    
    @abstractmethod
    def enable_rendering(self, base_pos: np.ndarray) -> None:
        """啟用可視化渲染，相機跟隨機器人"""
        pass
    
    @abstractmethod
    def update_camera(self, target_pos: np.ndarray) -> None:
        """更新相機位置以跟隨機器人"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PyBullet 實現 (延迟导入)
# ═══════════════════════════════════════════════════════════════════════════════

from pathlib import Path

# 嘗試導入 PyBullet (可能不可用)
try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    p = None
    pybullet_data = None


class PyBulletRobotSimulator(RobotInterface):
    """PyBullet 物理引擎的機器人模擬器實現"""

    def __init__(
        self,
        robot_description_name: str,
        physics_hz: float = 500,
        render: bool = False,
        gravity: float = -9.81,
        num_solver_iterations: int = 50,
    ):
        """
        初始化 PyBullet 模擬器

        Args:
            robot_description_name: robot_descriptions 套件中的機器人名稱
            physics_hz: 物理引擎頻率 (Hz)
            render: 是否啟用 GUI 渲染
            gravity: 重力加速度 (m/s²)
            num_solver_iterations: 求解器迭代次數
        """
        if not HAS_PYBULLET:
            raise RuntimeError(
                "PyBullet not installed. Install with: pip install pybullet"
            )

        self.robot_description_name = robot_description_name
        self.physics_hz = physics_hz
        self.render = render
        self.gravity = gravity
        self.num_solver_iterations = num_solver_iterations

        self._client: int = -1
        self._robot_id: int = -1
        self._plane_id: int = -1
        self._timestep = 1.0 / physics_hz
    
    def connect(self) -> None:
        """連線到 PyBullet"""
        if self.render:
            self._client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self._client)
        else:
            self._client = p.connect(p.DIRECT)
        self._setup_physics()

    def _setup_physics(self) -> None:
        """設置/恢復物理引擎參數（connect 和 reset 後都必須呼叫）"""
        p.setGravity(0, 0, self.gravity, physicsClientId=self._client)
        p.setTimeStep(self._timestep, physicsClientId=self._client)
        p.setPhysicsEngineParameter(
            numSolverIterations=self.num_solver_iterations,
            physicsClientId=self._client,
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    def disconnect(self) -> None:
        """斷開 PyBullet 連線"""
        if self._client >= 0:
            p.disconnect(physicsClientId=self._client)
            self._client = -1
    
    def reset(self, base_pos: np.ndarray, base_orn: np.ndarray) -> None:
        """重設模擬環境和機器人。

        第一次呼叫載入 URDF；後續只重置位置和速度，
        避免每個 episode 重新載入 URDF（主要效能瓶頸）。
        """
        if self._robot_id < 0:
            # 第一次：完整初始化
            p.resetSimulation(physicsClientId=self._client)
            self._setup_physics()

            self._plane_id = p.loadURDF(
                "plane.urdf", physicsClientId=self._client,
            )
            p.changeDynamics(
                self._plane_id, -1,
                lateralFriction=0.8,
                restitution=0.0,
                physicsClientId=self._client,
            )

            from robot_descriptions.loaders.pybullet import (
                load_robot_description,
            )
            self._robot_id = load_robot_description(
                self.robot_description_name,
                basePosition=base_pos.tolist(),
                baseOrientation=base_orn.tolist(),
                physicsClientId=self._client,
            )
        else:
            # 後續 episode：只重置位置和速度，不重載 URDF
            p.resetBasePositionAndOrientation(
                self._robot_id,
                base_pos.tolist(),
                base_orn.tolist(),
                physicsClientId=self._client,
            )
            p.resetBaseVelocity(
                self._robot_id,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                physicsClientId=self._client,
            )
    
    def step(self) -> None:
        """執行一單位物理模擬步"""
        p.stepSimulation(physicsClientId=self._client)
    
    # ── 觀測 ──────────────────────────────────────────────────────────────
    
    def get_base_position(self) -> np.ndarray:
        """獲取軀幹位置"""
        pos, _ = p.getBasePositionAndOrientation(
            self._robot_id,
            physicsClientId=self._client,
        )
        return np.array(pos, dtype=np.float32)
    
    def get_base_orientation_euler(self) -> np.ndarray:
        """獲取軀幹方向 (Euler)"""
        _, orn = p.getBasePositionAndOrientation(
            self._robot_id,
            physicsClientId=self._client,
        )
        euler = p.getEulerFromQuaternion(orn)
        return np.array(euler, dtype=np.float32)
    
    def get_base_linear_velocity(self) -> np.ndarray:
        """獲取軀幹線速度"""
        lin_vel, _ = p.getBaseVelocity(
            self._robot_id,
            physicsClientId=self._client,
        )
        return np.array(lin_vel, dtype=np.float32)
    
    def get_base_angular_velocity(self) -> np.ndarray:
        """獲取軀幹角速度"""
        _, ang_vel = p.getBaseVelocity(
            self._robot_id,
            physicsClientId=self._client,
        )
        return np.array(ang_vel, dtype=np.float32)
    
    def get_joint_positions(self, joint_indices: list) -> np.ndarray:
        """獲取關節位置"""
        positions = []
        for j in joint_indices:
            pos = p.getJointState(
                self._robot_id, j,
                physicsClientId=self._client,
            )[0]
            positions.append(pos)
        return np.array(positions, dtype=np.float32)
    
    def get_joint_velocities(self, joint_indices: list) -> np.ndarray:
        """獲取關節速度"""
        velocities = []
        for j in joint_indices:
            vel = p.getJointState(
                self._robot_id, j,
                physicsClientId=self._client,
            )[1]
            velocities.append(vel)
        return np.array(velocities, dtype=np.float32)
    
    def get_foot_contact(self, foot_link_indices: Dict[str, int]) -> Dict[str, bool]:
        """獲取腳部接觸狀態"""
        contacts = {}
        for foot_name, link_idx in foot_link_indices.items():
            contact_points = p.getContactPoints(
                self._robot_id,
                self._plane_id,
                linkIndexA=link_idx,
                physicsClientId=self._client,
            )
            contacts[foot_name] = len(contact_points) > 0
        return contacts
    
    # ── 設置 ──────────────────────────────────────────────────────────────
    
    def set_dynamics(
        self,
        link_index: int,
        mass: Optional[float] = None,
        lateral_friction: Optional[float] = None,
        restitution: Optional[float] = None,
    ) -> None:
        """設置 link 動力學參數"""
        kwargs = {"physicsClientId": self._client}
        if mass is not None:
            kwargs["mass"] = mass
        if lateral_friction is not None:
            kwargs["lateralFriction"] = lateral_friction
        if restitution is not None:
            kwargs["restitution"] = restitution
        
        p.changeDynamics(self._robot_id, link_index, **kwargs)
    
    # ── 渲染 ──────────────────────────────────────────────────────────────
    
    def enable_rendering(self, base_pos: np.ndarray) -> None:
        """啟用 GUI 並設置初始相機"""
        if not self.render:
            return
        
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=30,
            cameraPitch=-15,
            cameraTargetPosition=[base_pos[0], base_pos[1], base_pos[2]],
            physicsClientId=self._client,
        )
    
    def update_camera(self, target_pos: np.ndarray) -> None:
        """更新相機位置"""
        if not self.render:
            return
        
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=30,
            cameraPitch=-15,
            cameraTargetPosition=[target_pos[0], target_pos[1], 0.8],
            physicsClientId=self._client,
        )
    
    # ── PyBullet 特有方法 ────────────────────────────────────────────────
    
    def get_client_id(self) -> int:
        """獲取 PyBullet client ID (用於特殊操作)"""
        return self._client
    
    def get_robot_id(self) -> int:
        """獲取 PyBullet robot ID"""
        return self._robot_id
    
    def set_joint_motor_control(
        self,
        joint_indices: list,
        torques: np.ndarray,
    ) -> None:
        """
        設置關節扭矩控制
        
        Args:
            joint_indices: 關節索引列表
            torques: 扭矩值 array
        """
        for idx, j in enumerate(joint_indices):
            p.setJointMotorControl2(
                self._robot_id,
                j,
                controlMode=p.TORQUE_CONTROL,
                force=float(torques[idx]),
                physicsClientId=self._client,
            )
    
    def reset_joint_state(self, joint_pos_dict: Dict[int, float]) -> None:
        """重設關節到指定位置"""
        for j, angle in joint_pos_dict.items():
            p.resetJointState(
                self._robot_id,
                j,
                angle,
                targetVelocity=0.0,
                physicsClientId=self._client,
            )
    
    def disable_default_motors(self) -> None:
        """禁用所有預設馬達 (用於手動扭矩控制)"""
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client)
        for j in range(n_joints):
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self._client)
            if info[2] == 4:  # JOINT_FIXED
                continue
            p.setJointMotorControl2(
                self._robot_id,
                j,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self._client,
            )

    def lock_joints(self, joint_indices: list) -> None:
        """用高剛度位置控制鎖定指定關節在 0 位（如手臂/軀幹）"""
        for j in joint_indices:
            p.setJointMotorControl2(
                self._robot_id,
                j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                positionGain=0.5,
                velocityGain=0.1,
                force=300.0,
                physicsClientId=self._client,
            )

    def get_num_joints(self) -> int:
        """獲取關節總數"""
        return p.getNumJoints(self._robot_id, physicsClientId=self._client)

    def get_joint_name(self, joint_index: int) -> str:
        """獲取關節的 child link 名稱"""
        info = p.getJointInfo(
            self._robot_id, joint_index, physicsClientId=self._client
        )
        return info[12].decode()
