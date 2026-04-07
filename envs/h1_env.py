"""
Unitree H1 人形機器人環境 (簡化版，使用 WalkingEnv 基類)

使用robot_descriptions 套件加載 H1 URDF
"""

from typing import Optional
import numpy as np

from config import RobotConfig
from simulators import PyBulletRobotSimulator
from .base_walking_env import WalkingEnv


class H1Env(WalkingEnv):
    """Unitree H1 人形行走環境"""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        render_mode: Optional[str] = None,
    ):
        if robot_config is None:
            from config import ConfigManager
            mgr = ConfigManager()
            robot_config = mgr.load_robot_config("h1")

        simulator = PyBulletRobotSimulator(
            robot_description_name=robot_config.robot_description_name or "h1_description",
            physics_hz=robot_config.physics.physics_hz,
            render=render_mode == "human",
            gravity=robot_config.physics.gravity,
            num_solver_iterations=robot_config.physics.num_solver_iterations,
        )

        super().__init__(robot_config, simulator, render_mode)
        self._last_torques = np.zeros(len(robot_config.active_joints), dtype=np.float32)

    def _get_base_position(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.robot_config.spawn_z], dtype=np.float32)

    def _get_base_orientation(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _init_robot(self) -> None:
        """初始化機器人：腳底摩擦、站立姿態、馬達模式、鎖手臂"""
        foot_indices = self.robot_config.feet_link_indices
        self.robot_simulator.set_dynamics(
            foot_indices['left'],
            lateral_friction=self.robot_config.foot_friction,
            restitution=0.0,
        )
        self.robot_simulator.set_dynamics(
            foot_indices['right'],
            lateral_friction=self.robot_config.foot_friction,
            restitution=0.0,
        )

        if isinstance(self.robot_simulator, PyBulletRobotSimulator):
            self.robot_simulator.reset_joint_state(self.robot_config.stand_pose)
            # 關閉所有預設馬達（切換為扭矩控制）
            self.robot_simulator.disable_default_motors()
            # 鎖定手臂/軀幹關節 (joints 10-18)
            if self.robot_config.lock_joints:
                self.robot_simulator.lock_joints(self.robot_config.lock_joints)

    def _settle(self) -> None:
        settle_steps = int(0.3 * self.robot_config.physics.physics_hz)
        for _ in range(settle_steps):
            self.robot_simulator.step()

    def _apply_action(self, action: np.ndarray) -> None:
        """應用標準化扭矩動作(僅腿部 10 個關節)"""
        torques = np.clip(action, -1.0, 1.0) * np.array(self.robot_config.max_torque)
        self._last_torques = torques

        if isinstance(self.robot_simulator, PyBulletRobotSimulator):
            self.robot_simulator.set_joint_motor_control(
                self.robot_config.active_joints,
                torques,
            )

    def _get_last_torques(self) -> np.ndarray:
        return self._last_torques

    def _observe(self) -> np.ndarray:
        """觀測: torso(9) + joint_pos(10) + joint_vel(10) + contacts(2) = 31"""
        pos = self.robot_simulator.get_base_position()
        orn_euler = self.robot_simulator.get_base_orientation_euler()
        lin_vel = self.robot_simulator.get_base_linear_velocity()
        ang_vel = self.robot_simulator.get_base_angular_velocity()

        torso = np.concatenate([
            [pos[2]],
            orn_euler[[1, 0]],  # pitch, roll
            lin_vel,
            ang_vel,
        ])

        joint_pos = self.robot_simulator.get_joint_positions(
            self.robot_config.active_joints
        )
        joint_vel = self.robot_simulator.get_joint_velocities(
            self.robot_config.active_joints
        )

        foot_contact = self.robot_simulator.get_foot_contact(self.robot_config.feet_link_indices)
        contacts = np.array([
            float(foot_contact.get('left', False)),
            float(foot_contact.get('right', False)),
        ])

        obs = np.concatenate([torso, joint_pos, joint_vel, contacts])
        return obs.astype(np.float32)
